import numpy as np
from scipy.special import factorial as sp_factorial, erfinv
from matplotlib import pyplot as plt

import logging, os

from .HierarchicalModelDefinition import HierarchicalModel

# from ultranest.plot import cornerplot

from .placefield_detection import *
from .utils import *
import ultranest
from ultranest.popstepsampler import PopulationSliceSampler, generate_region_oriented_direction

from dynesty import NestedSampler, DynamicNestedSampler, pool as dypool, utils, plotting as dyplot

os.environ['OMP_NUM_THREADS'] = '1'
ultranest.__version__

logging.basicConfig(level=logging.INFO)

from scipy.ndimage import binary_opening, gaussian_filter1d as gauss_filter



class HierarchicalBayesModel(HierarchicalModel):

  ### possible speedup through...
  ###   parallelization of code

    def __init__(self, N, dwelltime, x_arr, f=1,hierarchical=[], wrap=[], logLevel=logging.ERROR):
        #self.N = gauss_filter(N,2,axis=1)
        self.N = N[np.newaxis,...]
        self.dwelltime = dwelltime[np.newaxis,...]
        self.nSamples, self.nbin = N.shape
        self.x_arr = x_arr[np.newaxis,np.newaxis,:]
        self.x_max = x_arr.max()
        self.Nx = len(x_arr)

        self.hierarchical = hierarchical
        self.wrap = wrap
                
        self.f = f

        if f>=1:
            self.set_priors(None,f,hierarchical,wrap)
        else:
            ## meaning: only noise model
            halfnorm_ppf = lambda x, loc, scale: loc + scale * np.sqrt(2) * erfinv(x)        
            norm_ppf = lambda x, loc, scale: loc + scale * np.sqrt(2) * erfinv(2*x - 1)
        
            self.set_priors({
                'A0': {
                    'hierarchical': {
                        'params':      {'loc':'mean', 'scale':'sigma'},
                        'function':    norm_ppf,
                    },
                    'mean': {
                        'params':       {'loc':0., 'scale':5},
                        'function':     halfnorm_ppf,
                    },
                    'sigma': {
                        'params':       {'loc':0., 'scale':2},
                        'function':     halfnorm_ppf,
                    },
                }},
                f,
                hierarchical,wrap)
                
        self.min_trials = 3
        
        self.log = logging.getLogger("nestLogger")
        self.log.setLevel(logLevel)

    
    def set_priors(self,priors_init=None,f=None,hierarchical=[],wrap=[]):
        
        halfnorm_ppf = lambda x, loc, scale: loc + scale * np.sqrt(2) * erfinv(x)        
        norm_ppf = lambda x, loc, scale: loc + scale * np.sqrt(2) * erfinv(2*x - 1)

        self.f = f if f is not None else self.f
        
        if priors_init is None:
            self.priors_init = {
                'A0': {
                    'hierarchical': {
                        'params':      {'loc':'mean', 'scale':'sigma'},
                        'function':    norm_ppf,
                    },
                    'mean': {
                        'params':       {'loc':0., 'scale':5},
                        'function':     halfnorm_ppf,
                    },
                    'sigma': {
                        'params':       {'loc':0., 'scale':2},
                        'function':     halfnorm_ppf,
                    },
                },
            }

            for f in range(1,self.f+1):
                self.priors_init[f'PF{f}_A'] = {
                        # maybe get rid of this parameter and just choose the best fitting height for each trial?!
                        # enable 2 mode fitting... (maybe just find best fit for location +/- 2sigma, instead of for whole trace?)
                        'hierarchical': {
                            'params':       {'loc':'mean', 'scale':'sigma'},
                            'function':     norm_ppf,
                        },
                        'mean': {
                            'params':       {'loc':0, 'scale':50},
                            'function':     halfnorm_ppf,
                        },
                        'sigma': {
                            'params':       {'loc':0., 'scale':5},
                            'function':     halfnorm_ppf,
                        },
                    }
                self.priors_init[f'PF{f}_sigma'] = {
                        'hierarchical': {
                            'params':       {'loc':'mean', 'scale':'sigma'},
                            'function':     norm_ppf
                        },
                        'mean': {
                            'params':       {'loc': 0.5, 'scale': 2},
                            'function':     halfnorm_ppf,
                        },
                        'sigma': {
                            'params':       {'loc':0., 'scale':2.},
                            'function':     halfnorm_ppf,
                        },
                    }
                self.priors_init[f'PF{f}_theta'] = {
                        'hierarchical': {
                            'params':       {'loc':'mean', 'scale':'sigma'},
                            'function':     norm_ppf,
                        },
                        'mean': {
                            'params':       {'stretch': self.Nx},
                            'function':     lambda x,stretch: x*stretch,
                        },
                        'sigma': {
                            'params':       {'loc': 0, 'scale': 3},
                            'function':     halfnorm_ppf,
                        },
                    }
                # self.priors_init[f'PF{f}_p'] = {
                #     'hierarchical': {
                #             'params':       {'loc':'mean', 'scale':'sigma'},
                #             # 'function':     norm_ppf,
                #             'function':     lambda x,loc,scale: x*2-1,
                #         },
                #         'mean': {
                #             'params':       {'stretch': 2.},
                #             'function':     lambda x,stretch: x*stretch-stretch/2.,
                #         },
                #         'sigma': {
                #             'params':       {'loc': 0, 'scale': 0.5},
                #             'function':     halfnorm_ppf,
                #         },
                # }
        else:
            self.priors_init = priors_init
        super().set_priors(self.priors_init,hierarchical,wrap)

    
    def tuning_curve(self,params, fields: int | None = None):

        ## build tuning-curve model
        mean_model_field = np.ones((1,self.nSamples,self.Nx))*params['A0']

        shift = self.x_max/2
        fields = params['PF'] if fields is None else [params['PF'][fields]]
        for field in fields: 
            mean_model_field += field['A']*np.exp(
                -(np.mod(self.x_arr - field['theta'] + shift,self.x_max)-shift)**2/(2*field['sigma']**2)
            )
        return mean_model_field
    

    def from_p_to_params(self,p_in):

        '''
            transform p_in to parameters for the model
        '''
        params = {}

        if self.f>0:
            params['PF'] = []
            for _ in range(self.f):
                params['PF'].append({})
        
        for key in self.priors:
            if self.priors[key]['meta']: continue

            if key.startswith('PF'):
                nField,key_param = key.split('_')
                nField = int(nField[2:])-1
                params['PF'][nField][key_param] = p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n'],np.newaxis]
            else:
                key_param = key.split('__')[0]
                params[key_param] = p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n'],np.newaxis]
            
        return params


    def set_logl_func(self,vectorized=True):

        '''
            TODO:
                instead of finding correlated trials before identification, run bayesian 
                inference on all neurons, but adjust log-likelihood:

                    * take placefield position, width, etc as hierarchical parameter 
                        (narrow distribution for location and baseline activity?)
                    * later, calculate logl for final parameter set to obtain active-trials (better logl)
                
                make sure all of this runs kinda fast!

                check, whether another hierarchy level should estimate noise-distribution parameters for overall data 
                    (thus, running inference only once on complete session, with N*4 parameters)
        '''
        
        self.fields = np.zeros((self.f,self.nSamples))
        def get_logl(p_in,plt_bool=False):
            
            if len(p_in.shape)==1:
                p_in = p_in[np.newaxis,:]
            N_in = p_in.shape[0]
            
            ## get proper parameters from p_in

            params = self.from_p_to_params(p_in)
            dParams_trial_from_total = {}
            if self.f>0:
                dParams_trial_from_total['PF'] = [{}]*self.f
            
            for key in self.priors:
                
                if self.priors[key]['n']>1:

                    if key.endswith('_p'): continue
                    
                    if key.startswith('PF'):
                        dParams_trial_from_total['PF'][int(key[2])-1][key.split('_')[1]] = (p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']] - p_in[:,[self.priors[key]['idx_mean']]])/p_in[:,[self.priors[key]['idx_sigma']]]
                    else:
                        dParams_trial_from_total[key] = (p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']] - p_in[:,[self.priors[key]['idx_mean']]])/p_in[:,[self.priors[key]['idx_sigma']]]
                
                # self.log.debug((f'{key_param}:',np.squeeze(params[key_param])))
                # self.log.debug((f'{key_param}:',params[key_param].shape))
            
            # print(dParams_trial_from_total)
            self.log.debug(('dParams:',dParams_trial_from_total))
            
            logp = np.zeros(N_in)
            
            mean_model_nofield = np.ones((1,self.nSamples,self.Nx))*params['A0']
            logp_nofield_at_position = poisson_spikes_log(mean_model_nofield,self.N,self.dwelltime)
            logp_nofield_at_position[np.isnan(logp_nofield_at_position)] = -10
            logp_nofield_trials = np.nansum(logp_nofield_at_position,axis=2)
            BIC_nofield = 1 * np.log(self.Nx) - 2 * logp_nofield_trials
                
            if self.f>0:

                # for each place field, identify whether it provides additional 
                # explanation wrt the nofield model by calculating the BIC in each trial
                # 
                # finally, for each trial, calculate the log-likelihood of the overall
                # model and - if placefields are combined, check whether it provides additional explanation

                field_in_trial = np.zeros((self.f,N_in,self.nSamples),dtype=bool)
                logp_field_trials = np.zeros((self.f,N_in,self.nSamples))

                discount_factor = 10.  ## don't like the hard-coded value here - any better idea?
                ## for all non-field-trials, introduce "pull" towards 0 for all parameters to avoid flat posterior
                zeroing_penalty = np.zeros(N_in)
                ### for all field-trials, enforce centering of parameters around meta parameter
                centering_penalty = np.zeros(N_in)

                ## iterate through each model
                for f in range(self.f): # achtually don't need the "if" before

                    mean_model_field = self.tuning_curve(params,f)
                    
                    ## get probability to observe N spikes (amplitude) within dwelltime for each bin in each trial
                    logp_field_at_position = poisson_spikes_log(mean_model_field,self.N,self.dwelltime)

                    ## fix 0-entries in probabilities (should maybe be done)
                    logp_field_at_position[np.isnan(logp_field_at_position)] = -10
                    
                    ## calculate trial-wise log-likelihoods for both models
                    logp_field_trials[f,...] = np.nansum(logp_field_at_position,axis=2)

                    # consider trials to be place-coding, when bayesian information 
                    # criterion (BIC) is lower than nofield-model. Number of parameters 
                    # for each trial is 1 (no field) vs 4 (single field)
                    BIC_field = 4 * np.log(self.Nx) - 2 * logp_field_trials[f,...]
                    
                    field_in_trial[f,...] = BIC_field < BIC_nofield

                    # print('field in trial',field_in_trial.shape)
                    # print(params)
                    # for each trial, add |p_t|/sum(|p_t|) * nofield_trial if p_t < 0, else |p_t|/sum(|p_t|) * 
                    

                    # if 'PF1_p' in self.priors:
                    #     field_in_trial = params['PF'][0]['p'][...,0] > 0
                    # else:

                    for key in dParams_trial_from_total['PF'][f]:
                        # if self.priors[key]['n']>1:
                            # if key.endswith('_p'): continue
                            # print(key,dParams_trial_from_total[key].shape)
                            # print(field_in_trial.shape)
                        zeroing_penalty += discount_factor*((dParams_trial_from_total['PF'][f][key]*(~field_in_trial[f,...]))**2).sum(axis=1)
                        centering_penalty += discount_factor*((dParams_trial_from_total['PF'][f][key]*field_in_trial[f,...]).sum(axis=1))**2

                    self.log.debug(('penalty (zeroing):',zeroing_penalty))
                    self.log.debug(('penalty (centering):',centering_penalty))
                    # print(f,'zeroing penalty:',zeroing_penalty)
                    # print(f,'centering penalty:',centering_penalty)
                

                ## calculate logp with all fields active
                mean_model_fields = self.tuning_curve(params)    
                logp_fields_at_position = poisson_spikes_log(mean_model_fields,self.N,self.dwelltime)
                logp_fields_at_position[np.isnan(logp_fields_at_position)] = -10
                logp_fields_trials = np.nansum(logp_fields_at_position,axis=2)

                self.fields += field_in_trial.sum(axis=1)


                '''
                    idea is: estimate 'activation' parameter, which is the probability of a trial to be place-coding
                    for each trial, calculate BIC for field and nofield model and calculate difference
                    the 'activation*nTrial' trials with the highest (positive) difference are considered to be place-coding
                '''

                self.log.debug(("field in trial",field_in_trial,field_in_trial.shape))
                
                # only consider the cell to be place coding, if at least min_trials are place-coding
                # if 'PF1_p' in self.priors:
                #     p_tmp = np.abs(params['PF'][0]['p'][...,0])+0.5
                #     p0_sum = p_tmp.sum(axis=-1)
                for i in range(N_in):
                    self.log.debug((f'logp trials {i}:',logp_nofield_trials[i,:]))

                    # print('field in trial:')
                    # print(field_in_trial[:,i,:])
                    # if field_in_trial[i,:].sum()>=self.min_trials:
                    
                    # if 'PF1_p' in self.priors:
                    #     logp[i] = (p_tmp[i,~field_in_trial[i,:]]/p0_sum[i] * logp_nofield_trials[i,~field_in_trial[i,:]]).sum()
                    #     logp[i] += (p_tmp[i,field_in_trial[i,:]]/p0_sum[i] * logp_field_trials[i,field_in_trial[i,:]]).sum()
                    # else:

                    logp[i] = logp_nofield_trials[i,np.all(~field_in_trial[:,i,:],axis=0)].sum()
                    # print('no field active:')
                    # print(np.all(~field_in_trial[:,i,:],axis=0))

                    fields_in_trial = np.all(field_in_trial[:,i,:],axis=0)
                    # print('all fields active:')
                    # print(fields_in_trial)
                    # print(field_in_trial[:,i,:]) 
                    logp[i] += logp_fields_trials[i,fields_in_trial].sum()


                    for f in range(self.f):
                        single_field_in_trial = field_in_trial[f,i,:] & ~fields_in_trial
                        # print(f'single field ({f}):')
                        # print(single_field_in_trial)
                        logp[i] += logp_field_trials[f,i,single_field_in_trial].sum()

                    # logp[i] = logp_nofield_trials[i,~field_in_trial[i,:]].sum()
                    # logp[i] += logp_field_trials[i,field_in_trial[i,:]].sum()
                    # else:
                    #     logp[i] = logp_nofield_trials[i,:].sum()
                    
                    logp[i] -= zeroing_penalty[i]
                    logp[i] -= centering_penalty[i]
                    self.log.debug((f'logp sum (with discount) {i}:',logp[i]))
            else:
                logp = np.nansum(logp_nofield_at_position,axis=(1,2))
                # logp[i] = logp_trials.sum(axis=1)
                
            if plt_bool:
                fig,ax = plt.subplots(3,N_in,figsize=(10,5))
                if len(ax)==1:
                    ax = ax[np.newaxis,:]

                ax[0][0].plot(self.x_arr[0,0,:],self.dwelltime[0,...].T)
                ax[0][1].plot(self.x_arr[0,0,:],self.N[0,...].T)
                ax[0][2].plot(self.x_arr[0,0,:],self.N[0,...].mean(axis=0))

                for i in range(N_in):
                    
                    ax[1][i].axvline(p_in[i,self.priors[key]['idx']],color='k',linestyle='--')

                    
                    ax[1][i].plot(self.x_arr[0,0,:],mean_model_nofield[i,...].T,linewidth=0.3)
                    ax[1][i].plot(self.x_arr[0,0,:],mean_model_field[i,...].T,linewidth=0.3)
                    
                    ax[1][i].plot(self.x_arr[0,0,:],mean_model_nofield[i,~field_in_trial[i,:],:].T,linewidth=0.5)
                    ax[1][i].plot(self.x_arr[0,0,:],mean_model_field[i,field_in_trial[i,:],:].T,linewidth=1.5)


                    ax[2][i].plot(self.x_arr[0,0,:],logp_nofield_at_position[i,~field_in_trial[i,...],:].T,linewidth=0.5)
                    ax[2][i].plot(self.x_arr[0,0,:],logp_field_at_position[i,field_in_trial[i,...],:].T,linewidth=1.5)

                plt.show()
            
            if vectorized:
                return logp
            else:
                return logp[0]

        return get_logl



def call_HBM(pathSession='../data/579ad/Session10',neuron=0,f=1,hierarchical=[],wrap=[],run_it=False,use_dynesty=False,logLevel=logging.ERROR):

    pathBehavior = os.path.join(pathSession,'aligned_behavior.pkl')


    pathActivity = [os.path.join(pathSession,file) for file in os.listdir(pathSession) if file.startswith('results_CaImAn')][0]
    # pathActivity = os.path.join(pathSession,'OnACID_results.hdf5')

    ld = load_dict_from_hdf5(pathActivity)
    #S = gauss_filter(ld['S'][neuron,:],2)
    S = ld['S'][neuron,:]


    with open(pathBehavior,'rb') as f_open:
        ld = pickle.load(f_open)

    nbin = 40
    bin_array = np.linspace(0,nbin-1,nbin)
    behavior = prepare_behavior(ld['time'],ld['position'],ld['reward_location'],nbin=nbin,f=15)
    activity = prepare_activity(S,behavior['active'],behavior['trials'],nbin=nbin)

    #print((S>0).sum(),activity['S'].sum())
    plt.figure()
    plt.plot(activity['trial_map'].T)
    plt.show(block=False)

    firingstats = get_firingstats_from_trials(activity['trial_map'],behavior['trials']['dwelltime'],N_bs=1000)

    fig = plt.figure()
    ax = fig.add_subplot(121)   
    ax.plot(behavior['time_raw'],activity['S'],'r',linewidth=0.3)

    ax = fig.add_subplot(122)
    ax.bar(bin_array,firingstats['map'],facecolor='b',width=1,alpha=0.2)
    # ax.bar(self.para['bin_array'],fmap,facecolor='r',width=1,alpha=0.2)
    ax.errorbar(bin_array,firingstats['map'],firingstats['CI'],ecolor='r',linestyle='',fmt='',elinewidth=0.3)
    plt.draw()
    plt.show(block=False)
    # time.sleep(1)
    # return

    # #%matplotlib nbagg 
    # fig,ax = plt.subplots(1,2)
    # ax[0].plot(ld['time'],S)
    # ax[0].scatter(behavior['time'],activity['S'],s=5,color='tab:orange')
    # ax[1].plot(activity['trial_map'].T)
    # plt.show(block=False)

    hbm = HierarchicalBayesModel(
        activity['trial_map'],
        behavior['trials']['dwelltime'],
        np.arange(nbin),
        f=f,
        hierarchical=hierarchical,
        logLevel=logLevel
    )

    if not run_it:
        return hbm, None, None


    
    if use_dynesty:
        # my_prior_transform = lambda p_in : hbm.transform_p(p_in,vectorized=False)
        my_prior_transform = hbm.set_prior_transform(vectorized=False)
        my_likelihood = hbm.set_logl_func(vectorized=False)
        print('running nested sampling')
        # print(np.where(hbm.pTC['wrap'])[0])
        with dypool.Pool(8,my_likelihood,my_prior_transform) as pool:
            sampler = NestedSampler(pool.loglike,pool.prior_transform,hbm.nParams,
                    pool=pool,
                    nlive=100,
                    bound='multi',
                    # periodic=np.where(hbm.pTC['wrap'])[0],
                    sample='slice'
                )
            sampler.run_nested()

        sampling_result = sampler.results
        # print(sampling_result)
        return hbm, sampling_result, sampler
    else:
        # print(hbm.wrap)
        my_prior_transform = hbm.set_prior_transform(vectorized=True)
        my_likelihood = hbm.set_logl_func()

        print(hbm.paramNames)
        sampler = ultranest.ReactiveNestedSampler(
            hbm.paramNames, 
            my_likelihood, my_prior_transform,
            wrapped_params=hbm.wrap,
            vectorized=True,num_bootstraps=20,
            ndraw_min=512
        )

        sampler.stepsampler = PopulationSliceSampler(popsize=100,nsteps=10,
                generate_direction=generate_region_oriented_direction)

        # sampler.stepsampler = ultranest.stepsampler.SliceSampler(
        #     nsteps=20,
        #     generate_direction=ultranest.stepsampler.generate_cube_oriented_direction,
        # )
        # sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=hbm.nPars)#, adaptive_nsteps='move-distance')
        # num_samples = hbm.nPars*100
        num_samples = 100

        sampling_result = sampler.run(
            min_num_live_points=num_samples,
            max_iters=50000,cluster_num_live_points=20,max_num_improvement_loops=1,
            # show_status=True,viz_callback='auto')  ## ... and run it #max_ncalls=500000,(f+1)*100,
            show_status=True,viz_callback=False)  ## ... and run it #max_ncalls=500000,(f+1)*100,
            #region_class=ultranest.mlfriends.SimpleRegion(),

        return hbm, sampling_result, sampler


def analyze_results(BM,results,sampler):

    params = BM.from_p_to_params(np.array(results['posterior']['mean'])[np.newaxis,:])
    # print(params)
    # print(params['PF'][0]['p'].shape)
    fig = plt.figure(figsize=(12,8))
    for trial in range(BM.nSamples):
        ax = fig.add_subplot(5,5,trial+1)
        ax.plot(BM.N[0,trial,:]/BM.dwelltime[0,trial,:],'k')

        lw = np.abs(params['PF'][0]['p'][0,trial,0])*2 if 'PF1_p' in BM.priors else 1
        # isfield = params['PF'][0]['p'][0,trial,0] > 0 if 'PF1_p' in BM.priors else BM.fields[trial] > BM.fields.max()/2.

        isfield_none = np.all(~(BM.fields[:,trial] > BM.fields.max()/2.),axis=0)
        isfield_all = np.all(BM.fields[:,trial] > BM.fields.max()/2.,axis=0)

        
        if isfield_none:
            ax.plot(BM.x_arr[0,0,:],np.ones(BM.Nx)*params['A0'][0,0,:],'r--',linewidth=lw)
        if isfield_all:
            ax.plot(BM.x_arr[0,0,:],BM.tuning_curve(params)[0,trial,:],'r--',linewidth=lw)
        
        for f in range(BM.f):
            isfield_single = BM.fields[f,trial] > BM.fields.max()/2. and ~isfield_all
            if isfield_single:
                ax.plot(BM.x_arr[0,0,:],BM.tuning_curve(params,f)[0,trial,:],'r--',linewidth=lw)


        #     # ax.plot(results['samples'][key][:,trial],'r',alpha=0.5)
        # if isfield:
        #     ax.plot(BM.x_arr[0,0,:],BM.tuning_curve(params)[0,trial,:],'r--',linewidth=lw)
        # else:
        #     ax.plot(BM.x_arr[0,0,:],np.ones(BM.Nx)*params['A0'][0,0,:],'r--',linewidth=lw)
        ax.set_ylim([0,200])
    
    plt.show(block=False)




def poisson_spikes(nu,N,T_total):
    return np.exp(N*np.log(nu*T_total) - np.log(sp_factorial(N)) - nu*T_total)

def poisson_spikes_log(nu,N,T_total):
    
    return N*np.log(nu*T_total) - np.log(sp_factorial(N)) - nu*T_total
