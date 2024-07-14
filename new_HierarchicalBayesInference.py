import numpy as np
from scipy.special import factorial as sp_factorial, erfinv
from matplotlib import pyplot as plt

import logging, os

# from ultranest.plot import cornerplot

from .placefield_detection import *
from .utils import *
import ultranest
from dynesty import DynamicNestedSampler, pool as dypool, utils, plotting as dyplot

os.environ['OMP_NUM_THREADS'] = '1'
ultranest.__version__

logging.basicConfig(level=logging.INFO)

from scipy.ndimage import binary_opening, gaussian_filter1d as gauss_filter



class HierarchicalBayesModel:

  ### possible speedup through...
  ###   parallelization of code

    def __init__(self, N, dwelltime, x_arr, hierarchical=False, logLevel=logging.ERROR):
        #self.N = gauss_filter(N,2,axis=1)
        self.N = N[np.newaxis,...]
        self.dwelltime = dwelltime[np.newaxis,...]
        self.nTrials, self.nbin = N.shape
        self.x_arr = x_arr[np.newaxis,np.newaxis,:]
        self.x_max = x_arr.max()
        self.Nx = len(x_arr)
        
        self.hierarchical = hierarchical
        
        self.f = 1

        
        self.set_priors()
                
        self.min_trials = 3
        
        self.log = logging.getLogger("nestLogger")
        self.log.setLevel(logLevel)

    
    def set_priors(self):
        
        halfnorm_ppf = lambda x, loc, scale: loc + scale * np.sqrt(2) * erfinv(x)
        #halfnorm_ppf = sstats.halfnorm.ppf
        
        norm_ppf = lambda x, loc, scale: loc + scale * np.sqrt(2) * erfinv(2*x - 1)
        #norm_ppf = sstats.norm.ppf
        
        self.priors_init = {
            'A0': {
                'mean': {
                    'params':       {'loc':0., 'scale':2},
                    'function':     halfnorm_ppf,
                    'wrap':         False,    
                },
                'hierarchical': False,
                # 'hierarchical': {
                #     'n':           self.nTrials,
                #     'params':      {'loc':0, 'scale':1},
                #     'function':    norm_ppf,
                # } 
            },
            'A': {
                'mean': {
                    'params':       {'loc':5, 'scale':5},
                    'function':     halfnorm_ppf,
                    'wrap':         False,    
                },
                'hierarchical': False,
                # 'hierarchical': {
                #     'n':            self.nTrials,
                #     'params':       {'loc': 0, 'scale': 2},
                #     'function':     norm_ppf,
                # } 
            },
            
            'sigma': {
                'mean': {
                    'params':       {'loc': 0.2, 'scale': 1},
                    'function':     halfnorm_ppf,
                    'wrap':         False,    
                },
                'hierarchical': False,
                # 'hierarchical': {
                #     'n':            self.nTrials,
                #     'params':       {'loc':0, 'scale':1},
                #     'function':    norm_ppf
                # } 
            },
            'theta': {
                'mean': {
                    'params':       {'stretch': self.Nx},
                    'function':     lambda x,stretch: x*stretch,
                    'wrap':         True,    
                },
                'sigma': {
                    'params':       {'loc': 0, 'scale': 1},
                    'function':     halfnorm_ppf,
                    'wrap':         False,    
                },
                # 'hierarchical': False,
                'hierarchical': {
                    'params':       {'loc': 'mean', 'scale': 'sigma'},
                    'function':     norm_ppf,
                    'wrap':         True,    
                } 
            },
        }
        
        ## could create new array here...?
        self.paraNames = []
        self.priors = {}
        self.pTC = {}
        
        
        ct = 0
        for key in self.priors_init:
            for var in self.priors_init[key]:
                paraName = f"{key}_{var}"
                print(paraName)
                
                if not (var=='hierarchical'):
                    self.priors[paraName] = {}
                    
                    self.paraNames.append(paraName)
                    self.priors[paraName]['idx'] = ct
                    self.priors[paraName]['n'] = 1
                    self.priors[paraName]['meta'] = isinstance(self.priors_init[key]['hierarchical'],dict) and self.hierarchical
                    ct += 1

                    self.priors[paraName]['transform'] = \
                        lambda x,params=self.priors_init[key][var]['params'],fun=self.priors_init[key][var]['function']: fun(x,**params)

                    # if self.priors_init[key][var]['wrap']:
                        # self.pTC['wrap'][self.priors[paraName]['idx']] = True


                elif (var=='hierarchical') and self.hierarchical and self.priors_init[key]['hierarchical']:
                    self.priors[paraName] = {}
                    self.priors[paraName]['idx'] = ct
                    self.priors[paraName]['idx_mean'] = self.priors[f"{key}_{self.priors_init[key]['hierarchical']['params']['loc']}"]['idx']
                    self.priors[paraName]['idx_sigma'] = self.priors[f"{key}_{self.priors_init[key]['hierarchical']['params']['scale']}"]['idx']
                    self.priors[paraName]['n'] = self.nTrials
                    self.priors[paraName]['meta'] = False
                    for i in range(self.nTrials):
                        self.paraNames.append(f'{key}_{i}')

                        # if self.priors_init[key][var]['wrap']:
                            # self.pTC['wrap'][self.priors[paraName]['idx']+i] = True
                    ct += self.nTrials
                
                    self.priors[paraName]['transform'] = \
                        lambda x,params,fun=self.priors_init[key][var]['function']: fun(x,**params)
        
        self.nPars = len(self.paraNames)
        self.pTC['wrap'] = np.zeros(self.nPars).astype('bool')
        for key in self.priors:
            print(key)
            key_root, key_var = key.split('_')
            if 'hierarchical' in self.priors_init[key_root]:
                self.pTC['wrap'][self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']] = self.priors_init[key_root][key_var]['wrap']
        

    def transform_p(self,p_in,vectorized=True):
        
        if len(p_in.shape)==1:
            p_in = p_in[np.newaxis,...]
        p_out = np.zeros_like(p_in)
        
        for key in self.priors:
            
            if self.priors[key]['n']==1:
            
                p_out[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']] = self.priors[key]['transform'](p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']])

            else:
                params = {
                    'loc':          p_out[:,self.priors[key]['idx_mean'],np.newaxis],
                    'scale':        p_out[:,self.priors[key]['idx_sigma'],np.newaxis],
                    # 'loc':          0,
                    # 'scale':        1,
                }
                p_out[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']] = self.priors[key]['transform'](p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']],params=params)
        
        if vectorized:
            return p_out
        else:
            return p_out[0,:]

        
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
        
        self.fields = np.zeros(self.nTrials)
        def get_logl(p_in,plt_bool=False):
            
            if len(p_in.shape)==1:
                p_in = p_in[np.newaxis,:]
            N_in = p_in.shape[0]
            
            ## get proper parameters from p_in
            params = {}
            dParams_trial_from_total = {}
            # dParams_trial_from_total = np.zeros((N_in,self.nTrials))
            for key in self.priors:
                
                key_param = key.split('_')[0]
                if self.priors[key]['meta']: continue
                # print('continue from here....')


                params[key_param] = p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n'],np.newaxis]


                # mean_key = self.priors_init[root_var]['hierarchical']['params']['loc']
                # idx_mean = np.full((1,1) if vectorized else 1,
                #     self.priors[mean_key]['idx'])
                # if self.priors[key]['n']==1:

                if self.priors[key]['n']>1:

                    # params[key_param] = p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n'],np.newaxis] + p_in[:,[self.priors[key]['idx_mean']],np.newaxis]
                    
                    dParams_trial_from_total[key] = (p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']] - p_in[:,[self.priors[key]['idx_mean']]])/p_in[:,[self.priors[key]['idx_sigma']]]
                    # dParams_trial_from_total[key] = p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']]
                

                # dParams_trial_from_total += ((p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']]-self.priors[key]['hierarchical']['params']['loc'])/self.priors[key]['hierarchical']['params']['scale'])#**2
 
                if self.pTC['wrap'][self.priors[key]['idx']]:
                # key=='theta':
                    params[key_param] = np.mod(params[key_param],self.Nx)   ## wrap value shouldnt be hardcoded, but provided in setp_priors
                    # self.log.debug(('p_in:',p_in[:,self.priors[key]['idx']+1:self.priors[key]['idx']+1+self.nTrials]))

                # if self.priors[key]['hierarchical']:
                #     ## hierarchical parameters are computed from base value plus trial-dependent values
                #     params[key] = p_in[:,self.priors[key]['idx'],np.newaxis] + \
                #        p_in[:,self.priors[key]['idx']+1:self.priors[key]['idx']+1+self.nTrials]
                    
                #     ## theta is wrapped (circular environment)
                #     if key=='theta':
                #         params[key] = np.mod(params[key],self.Nx)
                #         self.log.debug(('p_in:',p_in[:,self.priors[key]['idx']+1:self.priors[key]['idx']+1+self.nTrials]))

                #     ## compute SD-deviations from base value
                #     ## this should be summed up separately for each parameter, to enable centering penalty
                #     dParams_trial_from_total += ((p_in[:,self.priors[key]['idx']+1:self.priors[key]['idx']+1+self.nTrials]-self.priors[key]['hierarchical']['params']['loc'])/self.priors[key]['hierarchical']['params']['scale'])#**2
                # else:
                    # params[key] = p_in[:,self.priors[key]['idx'],np.newaxis]
                
                # params[key_param] = params[key_param][...,np.newaxis]
                self.log.debug((f'{key_param}:',np.squeeze(params[key_param])))
                self.log.debug((f'{key_param}:',params[key_param].shape))
            
            
            self.log.debug(('dParams:',dParams_trial_from_total))
            
            ## build tuning-curve model
            mean_model_nofield = np.ones((1,self.nTrials,self.Nx))*params['A0']
            mean_model_field = np.ones((1,self.nTrials,self.Nx))*params['A0']
            if N_in > 1:
                for j in [-1,0,1]:
                    mean_model_field += \
                        (params['A']*np.exp(
                                -(self.x_arr - params['theta'] + self.x_max*j)**2/(2*params['sigma']**2)
                            )
                        )
            
            ## get probability to observe N spikes (amplitude) within dwelltime for each bin in each trial
            logp_nofield_at_position = poisson_spikes_log(mean_model_nofield,self.N,self.dwelltime)
            logp_field_at_position = poisson_spikes_log(mean_model_field,self.N,self.dwelltime)
            ## fix 0-entries in probabilities (should maybe be done)
            logp_nofield_at_position[np.isnan(logp_nofield_at_position)] = -1
            logp_field_at_position[np.isnan(logp_field_at_position)] = -1
            
            logp_trials = np.zeros((2,N_in,self.nTrials))
            logp_trials[0,...] = np.nansum(logp_nofield_at_position,axis=2)
            logp_trials[1,...] = np.nansum(logp_field_at_position,axis=2)

            # consider trials to be place-coding, when bayesian information criterion (BIC) 
            # is lower than nofield-model. Number of parameters for each trial is 1 vs 4, 
            # as trial dependent parameters only affect a single trial, each
            BIC_nofield = 1 * np.log(self.Nx) - 2 * logp_trials[0,...]
            BIC_field = 4 * np.log(self.Nx) - 2 * logp_trials[1,...]
            
            field_in_trial = BIC_field < BIC_nofield
            self.log.debug(("field in trial",field_in_trial,field_in_trial.shape))
            
            ## for all non-field-trials, introduce "pull" towards 0 for all parameters to avoid flat posterior
            zeroing_penalty = np.zeros(N_in)
            centering_penalty = np.zeros(N_in)
            for key in self.priors:
                if self.priors[key]['n']>1:
                    zeroing_penalty += ((dParams_trial_from_total[key]*(~field_in_trial))**2).sum(axis=1)
                    centering_penalty += ((dParams_trial_from_total[key]*field_in_trial).sum(axis=1))**2
            # zeroing_penalty = 10*(dParams_trial_from_total**2).sum(axis=1)        
            self.log.debug(('penalty (zeroing):',zeroing_penalty))
            self.log.debug(('penalty (centering):',centering_penalty))
            
            # print('zeroing: ',zeroing_penalty.mean(),', centering: ',centering_penalty.mean())
            self.fields += field_in_trial.sum(axis=0)
            
            
            # self.log.debug(~field_in_trial)
            
            # only consider the cell to be place coding, if at least min_trials are place-coding
            logp = np.zeros(N_in)
            for i in range(N_in):
                self.log.debug((f'logp trials {i}:',logp_trials[0,i,:]))#
                if field_in_trial[i,:].sum()>=self.min_trials:
                                
                    logp[i] = logp_trials[0,i,~field_in_trial[i,:]].sum()
                    logp[i] += logp_trials[1,i,field_in_trial[i,:]].sum()
                else:
                    logp[i] = logp_trials[0,i,:].sum()
                logp[i] -= zeroing_penalty[i]
                logp[i] -= centering_penalty[i]
                self.log.debug((f'logp sum (with discount) {i}:',logp[i]))
                
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



def call_HBM(pathSession='../data/556wt/Session12',neuron=0,hierarchical=True,run_it=False,use_dynesty=False,logLevel=logging.ERROR):

    pathBehavior = os.path.join(pathSession,'aligned_behavior.pkl')
    pathActivity = os.path.join(pathSession,'OnACID_results.hdf5')

    ld = load_dict_from_hdf5(pathActivity)
    #S = gauss_filter(ld['S'][neuron,:],2)
    S = ld['S'][neuron,:]


    with open(pathBehavior,'rb') as f_open:
        ld = pickle.load(f_open)

    nbin = 40
    bin_array = np.linspace(0,nbin-1,nbin)
    behavior = prepare_behavior(ld['time'],ld['position'],ld['reward_location'],nbin=nbin,f=15)
    activity = prepare_activity(S,behavior['active'],behavior['trials'],nbin=nbin)

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
        hierarchical=hierarchical,
        logLevel=logLevel
    )

    if not run_it:
        return hbm, None, None

    my_prior_transform = hbm.transform_p
    my_likelihood = hbm.set_logl_func()

    
    if use_dynesty:
        my_prior_transform = lambda p_in : hbm.transform_p(p_in,vectorized=False)
        my_likelihood = hbm.set_logl_func(vectorized=False)
        print('running nested sampling')
        print(np.where(hbm.pTC['wrap'])[0])
        with dypool.Pool(8,my_likelihood,my_prior_transform) as pool:
            sampler = DynamicNestedSampler(pool.loglike,pool.prior_transform,hbm.nPars,
                    pool=pool,
                    periodic=np.where(hbm.pTC['wrap'])[0],
                    sample='slice'
                )
            sampler.run_nested()

        sampling_result = sampler.results
        print(sampling_result)
        return hbm, sampling_result, sampler
    else:
        my_prior_transform = hbm.transform_p
        my_likelihood = hbm.set_logl_func()
        print(hbm.paraNames)
        sampler = ultranest.ReactiveNestedSampler(
            hbm.paraNames, 
            my_likelihood, my_prior_transform,
            wrapped_params=hbm.pTC['wrap'],
            vectorized=True,num_bootstraps=20,
            ndraw_min=512
        )


        # sampler.stepsampler = ultranest.stepsampler.SliceSampler(
        #     nsteps=20,
        #     generate_direction=ultranest.stepsampler.generate_cube_oriented_direction,
        # )
        num_samples = hbm.nPars*100

        sampling_result = sampler.run(
            min_num_live_points=num_samples,
            max_iters=20000,cluster_num_live_points=20,max_num_improvement_loops=3,
            show_status=True,viz_callback='auto')  ## ... and run it #max_ncalls=500000,(f+1)*100,
            #region_class=ultranest.mlfriends.SimpleRegion(),

        return hbm, sampling_result, sampler

    # sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=hbm.nPars)#, adaptive_nsteps='move-distance')
    
    return sampling_result, sampler



def tuning_curve(x_arr,p):

    """
        defines the model by providing the average expected firing rate at each bin
    """
    x_max = x_arr.shape[0]

    if len(p.shape)==1:
        p = p[np.newaxis,:]
    p = p[...,np.newaxis,np.newaxis]   ## adding 2 axes for 1. number of trials, 2. number of bins
    
    mean_model_nofield = np.ones((p.shape[0],x_max))*p[:,0,:,0]
    mean_model_field = np.ones((p.shape[0],p.shape[1]-4,x_max))*p[:,0,...]
    if p.shape[1] > 1:
        for j in [-1,0,1]:
            mean_model_field += \
                (p[:,1,...]*np.exp(
                        -(x_arr[np.newaxis,np.newaxis,:] - p[:,3,...] - p[:,4:,:,0] + x_max*j)**2/(2*p[:,2,...]**2)
                    )
                )

    return mean_model_nofield, mean_model_field

def poisson_spikes(nu,N,T_total):
    return np.exp(N*np.log(nu*T_total) - np.log(sp_factorial(N)) - nu*T_total)

def poisson_spikes_log(nu,N,T_total):
    # if nu==0:
    #     return -np.inf
    # nu[nu<=0] = 1e-5
    # print(T_total)
    # T_total[T_total==0] = 0.01
    
    return N*np.log(nu*T_total) - np.log(sp_factorial(N)) - nu*T_total



# hbm,sampler = call_HBM('../data/556wt/Session12',neuron=2,logLevel=logging.ERROR)