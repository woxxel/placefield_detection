import logging, time, os, warnings
import numpy as np
from scipy.special import factorial as sp_factorial, erfinv, erf
from scipy.ndimage import gaussian_filter1d as gauss_filter
from scipy.interpolate import interp1d


import ultranest
from ultranest.popstepsampler import PopulationSliceSampler, generate_region_oriented_direction
from ultranest.mlfriends import RobustEllipsoidRegion


from HierarchicalModelDefinition import HierarchicalModel
from utils import model_of_tuning_curve

os.environ['OMP_NUM_THREADS'] = '1'
logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings("ignore")

logger = logging.getLogger("ultranest")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)



class HierarchicalBayesModel(HierarchicalModel):

    def __init__(self, N, dwelltime, x_arr, logLevel=logging.ERROR):
        self.N = N[np.newaxis,...]
        self.dwelltime = dwelltime[np.newaxis,...]
        self.nSamples, self.nbin = N.shape
        self.x_arr = x_arr[np.newaxis,np.newaxis,:]
        self.x_max = x_arr.max()
        self.Nx = len(x_arr)

        self.firing_map = N.sum(axis=0) / dwelltime.sum(axis=0)

        ## pre-calculate log-factorial for speedup
        self.log_N_factorial = np.log(sp_factorial(self.N))
                
        self.log = logging.getLogger("nestLogger")
        self.log.setLevel(logLevel)

    
    def set_priors(self,priors_init=None,N_f=None,hierarchical_in=[],wrap=[]):
        
        halfnorm_ppf = lambda x, loc, scale: loc + scale * np.sqrt(2) * erfinv(x)        
        norm_ppf = lambda x, loc, scale: loc + scale * np.sqrt(2) * erfinv(2*x - 1)

        self.N_f = N_f
        
        if priors_init is None:

            A0_guess, A_guess = np.percentile(self.firing_map[self.firing_map>0],[10,90])
            self.priors_init = {
                'A0': {
                    'hierarchical': {
                        'params':      {'loc':'mean', 'scale':'sigma'},
                        'function':    norm_ppf,
                    },
                    'mean': {
                        # 'params':       {'loc':0., 'scale':50},
                        'params':       {'loc':0., 'scale':A0_guess},
                        'function':     halfnorm_ppf,
                    },
                    'sigma': {
                        'params':       {'loc':0., 'scale':A0_guess/10.},
                        'function':     halfnorm_ppf,
                    },
                },
            }

            for f in range(1,self.N_f+1):
                
                self.priors_init[f'PF{f}_A'] = {
                        # maybe get rid of this parameter and just choose the best fitting height for each trial?!
                        'hierarchical': {
                            'params':       {'loc':'mean', 'scale':'sigma'},
                            'function':     norm_ppf,
                        },
                        'mean': {
                            # 'params':       {'loc':0, 'scale':100},
                            'params':       {'loc':0, 'scale':A_guess},
                            'function':     halfnorm_ppf,
                        },
                        'sigma': {
                            'params':       {'loc':0., 'scale':A_guess/10.},
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
                            'params':       {'loc': 0, 'scale': 2},
                            'function':     halfnorm_ppf,
                        },
                    }
                
        else:
            self.priors_init = priors_init
        

        hierarchical = []
        for f in range(1,self.N_f+1):
            for h in hierarchical_in:
                hierarchical.append(f'PF{f}_{h}')

        super().set_priors(self.priors_init,hierarchical,wrap)

    
    def model_of_tuning_curve(self, params, fields: int | str | None = 'all',stacked=False):

        return model_of_tuning_curve(self.x_arr,params,self.Nx,self.nSamples,fields,stacked)
    

    def from_p_to_params(self,p_in):

        '''
            transform p_in to parameters for the model
        '''
        params = {}

        if self.N_f>0:
            params['PF'] = []
            for _ in range(self.N_f):
                params['PF'].append({})
        
        for key in self.priors:
            if self.priors[key]['meta']: continue

            if key.startswith('PF'):
                nField,key_param = key.split('_')
                nField = int(nField[2:])-1
                params['PF'][nField][key_param] = p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']]
            else:
                key_param = key.split('__')[0]
                params[key_param] = p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']]
            
        return params

    def timeit(self,msg=None):
        if not msg is None:# and (self.time_ref):
            self.log.debug(f'time for {msg}: {(time.time()-self.time_ref)*10**6}')
        
        self.time_ref = time.time()
        


    def set_logp_func(self,vectorized=True,penalties=['parameter_bias','reliability','overlap']):

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

        import matplotlib.pyplot as plt
        
        def get_logp(p_in,get_active_model=False,get_logp=False,get_tuning_curve=False):
            
            t_ref = time.time()
            ## adjust shape of incoming parameters to vectorized analysis
            if len(p_in.shape)==1:
                p_in = p_in[np.newaxis,:]
            N_in = p_in.shape[0]

            self.timeit()
            
            params = self.from_p_to_params(p_in)
            # print(params)
            self.timeit('transforming parameters')

            tuning_curve_models = self.model_of_tuning_curve(params,stacked=True)
            if get_tuning_curve:
                return tuning_curve_models
            self.timeit('tuning curve model')

            logp_at_trial_and_position = np.zeros((2**self.N_f,N_in,self.nSamples,self.Nx))
            logp_at_trial_and_position[0,...] = self.probability_of_spike_observation(tuning_curve_models[0,...])

            if self.N_f>0:
                if self.N_f>1:
                    for field_model in range(self.N_f):
                        logp_at_trial_and_position[field_model+1,...] = self.probability_of_spike_observation(tuning_curve_models[[0,field_model+1],...].sum(axis=0))
                
                ## also, calculate log-likelihood for all fields combined
                logp_at_trial_and_position[-1,...] = self.probability_of_spike_observation(tuning_curve_models.sum(axis=0))
                self.timeit('poisson')

                infield_range = self.generate_infield_ranges(params)
                self.timeit('infield ranges')

                AIC = self.compute_AIC(logp_at_trial_and_position,infield_range)
                self.timeit('AIC')
                

                active_model = self.obtain_active_model(AIC)
                if get_active_model:
                    return active_model
                self.timeit('active model')


                # print('AIC',AIC.shape)
                # print(params['A0'])
                # print(AIC[...,5])
                # print(logp_at_trial_and_position[[0,1,2],0,:,:])
                # plt.figure()
                # trial = 2
                # ax = plt.subplot(311)
                # ax.plot(logp_at_trial_and_position[0,0,trial,:],'r')
                # ax.plot(logp_at_trial_and_position[1,0,trial,:],'b--')
                # ax.plot(logp_at_trial_and_position[2,0,trial,:],'g--')
                # ax.plot(logp_at_trial_and_position[3,0,trial,:],'k--')
                
                # ax = plt.subplot(312)
                # ax.plot(logp_at_trial_and_position[0,0,...].sum(axis=-1),'r')
                # ax.plot(logp_at_trial_and_position[1,0,...].sum(axis=-1),'b--')
                # ax.plot(logp_at_trial_and_position[2,0,...].sum(axis=-1),'g--')
                # ax.plot(logp_at_trial_and_position[3,0,...].sum(axis=-1),'k--')

                # ax = plt.subplot(313)
                # ax.plot(0,logp_at_trial_and_position[0,0,...].sum(),'ro')
                # ax.plot(0,logp_at_trial_and_position[1,0,...].sum(),'bo')
                # ax.plot(0,logp_at_trial_and_position[2,0,...].sum(),'go')
                # ax.plot(0,logp_at_trial_and_position[3,0,...].sum(),'ko')

                # best_log = np.max(logp_at_trial_and_position[:,0,...].sum(axis=-1),axis=0).sum()
                # print(f"{best_log=}")
                # plt.show()

                # logp_nofield = logp_at_trial_and_position[0,0,...].sum(axis=-1)
                # logp_field1 = logp_at_trial_and_position[1,0,...].sum(axis=-1)
                # logp_field2 = logp_at_trial_and_position[2,0,...].sum(axis=-1)
                # dlogp1 = logp_field1 - logp_nofield

                # print(f"{dlogp1[~np.any(active_model[[1,3],0,...],axis=0)]=}")
                # print(dlogp1[~np.any(active_model[[1,3],0,...],axis=0)].sum())

                # dlogp2 = logp_field2 - logp_nofield
                # print(f"{dlogp2[~np.any(active_model[[2,3],0,...],axis=0)]=}")
                # print(dlogp2[~np.any(active_model[[2,3],0,...],axis=0)].sum())

            else:
                active_model = np.ones((1,N_in,self.nSamples),'bool')
                infield_range = None

            if get_logp:
                return logp_at_trial_and_position,active_model

            logp = np.sum(logp_at_trial_and_position,where=active_model[...,np.newaxis],axis=(3,2,0))   
            self.timeit('raw logp')

            self.log.debug((f'{logp=}'))

            if self.N_f > 0:
                logp -= self.calculate_logp_penalty(p_in,params,logp_at_trial_and_position,active_model,infield_range,penalties)
                self.timeit('penalties')
            self.log.debug((f'{logp=}'))
            
            if vectorized:
                return logp
            else:
                return logp[0]

        return get_logp
    
    
    def generate_infield_ranges(self,params,cut_range=2.):
        ## define ranges, in which the different models are compared
        N_in = params['A0'].shape[0]
        infield_range = np.zeros((self.N_f,N_in,self.nSamples,self.Nx),dtype=bool)

        for field_model,PF in enumerate(params['PF']): # actually don't need the "if" before

            lower = np.floor(np.mod(PF['theta'] - cut_range*PF['sigma'],self.Nx)).astype('int')
            upper = np.ceil(np.mod(PF['theta'] + cut_range*PF['sigma'],self.Nx)).astype('int')

            for i in range(N_in):
                for trial in range(self.nSamples):
                    if lower[i,trial] < upper[i,trial]:
                        infield_range[field_model,i,trial,lower[i,trial]:upper[i,trial]] = True
                    else:
                        infield_range[field_model,i,trial,lower[i,trial]:] = True
                        infield_range[field_model,i,trial,:upper[i,trial]] = True
        return infield_range

    

    def obtain_active_model(self,AIC):
        '''
            something seems off with how area is defined and used...
            which axis should I apply reliability penalty to? how is it interpreted, if f(i,j) is the off-diagonal field? ...
        '''
        ## entry 0 should be nofield model and all possible combinations of field models
        ## I'm actually not quite sure why it evolves to f**2, but it holds
        ## now, how do I properly assign IDs to to model (especially combinations)?

        N_in = AIC.shape[1]
        active_model_reference = np.argmin(AIC,axis=0)

        active_model = np.zeros((2**self.N_f,N_in,self.nSamples),dtype=bool)

        active_model[0,...] = np.all(active_model_reference==0, axis=1)
        if self.N_f==1:
            active_model[1,...] = np.any(active_model_reference==1, axis=1)

        if self.N_f==2:
            active_model[1,...] = np.any(active_model_reference==1, axis=1) & np.all(active_model_reference!=2, axis=1)
            active_model[2,...] = np.any(active_model_reference==2, axis=1) & np.all(active_model_reference!=1, axis=1)
            active_model[-1,...] = np.all(~active_model[:-1,...], axis=0)

        return active_model

    
    def compute_AIC(self,logp_at_trial_and_position,infield_range):
        N_in = logp_at_trial_and_position.shape[1]

        AIC = np.zeros((self.N_f+1,N_in,self.N_f,self.nSamples))
        for field_area in range(self.N_f):

            nDatapoints = infield_range[field_area,...].sum(axis=-1)

            ## calculate trial-wise log-likelihoods for both models
            logp_field_trials = np.sum(
                logp_at_trial_and_position,
                where=infield_range[[field_area],...],
                axis=-1)

            for field_model in range(self.N_f + 1):
                '''
                    consider trials to be place-coding, when Akaike information 
                    criterion (AIC) is lower than nofield-model. Number of parameters 
                    for each trial is 1 (no field) vs 4 (single field)
                '''
                #off_field = (field_model>0) and (field_model != (field_area+1))
                nParameter = 1 + 3*(field_model>0)# + 3*off_field
                # print(f'{field_area=}, {field_model=}: {nParameter=}')

                AIC[field_model,:,field_area,:] = nParameter * np.log(nDatapoints) - 2 * logp_field_trials[field_model,...]
        
        return AIC

        
    def calculate_logp_penalty(self,p_in,params,logp_at_trial_and_position,active_model,infield_range,penalties=['parameter_bias','reliability','overlap'],penalty_factor=None,no_go_factor=10**6):
        '''
            calculates several penalty values for the log-likelihood to adjust inference
                - zeroing_penalty: penalizes parameters to be far from 0

                - centering_penalty: penalizes parameters to be far from meta-parameter

                - activation_penalty: penalizes trials to be place-coding 
                        This could maybe introduce AIC as penalty factor, to bias towards non-coding, and only consider it to be better, when it surpasses AIC? (how to properly implement?)
            
        '''

        if not penalty_factor:
            ## choose the penalty to be equal to the AIC difference between nofield and field model
            penalty_factor = 3. * np.log(self.Nx)
        
        N_in = p_in.shape[0]
        
        if self.N_f>1:
            active_model[1,active_model[-1,...]] = True
            active_model[2,active_model[-1,...]] = True

        zeroing_penalty = np.zeros(N_in)
        centering_penalty = np.zeros(N_in)

        if 'parameter_bias' in penalties:
            ## for all non-field-trials, introduce "pull" towards 0 for all parameters to avoid flat posterior
            ### for all field-trials, enforce centering of parameters around active meta parameter

            for key in self.priors:
                
                if key.startswith('PF'):
                    f = int(key[2])
                
                ## hierarchical parameters fluctuate around meta-parameters, and should be centered around them, as well as bias towards them for non-field trials
                if self.priors[key]['n']>1:
                    
                    dParam_trial_from_total = (p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']] - p_in[:,[self.priors[key]['idx_mean']]])/p_in[:,[self.priors[key]['idx_sigma']]]
                    
                    zeroing_penalty += penalty_factor * ((dParam_trial_from_total*(~active_model[f,...]))**2).sum(axis=1)
                    centering_penalty += penalty_factor * ((dParam_trial_from_total*active_model[f,...]).sum(axis=1))**2
            self.timeit('zeroing/centering penalty')


        overlap_penalty = np.zeros(N_in)
        if ('overlap' in penalties) and (self.N_f > 1):
            overlap_range = np.all(infield_range,axis=0).sum(axis=-1)
            for PF in params['PF']:
                overlap_penalty += penalty_factor*norm_cdf(overlap_range-2*PF['sigma'],0,PF['sigma']).sum(axis=-1)
            self.timeit('overlap penalty')
        
        reliability_penalty = np.zeros(N_in)
        if 'reliability' in penalties:
            
            dlogp = np.zeros((self.N_f,N_in))
            logp_nofield = logp_at_trial_and_position[0,...].sum(axis=-1)
            for f in range(self.N_f):
                logp_field = logp_at_trial_and_position[f+1,...].sum(axis=-1)
                dlogp_trials = np.maximum(logp_nofield - logp_field,0)
                dlogp[f,:] = np.sum(dlogp_trials,where=~np.any(active_model[[f+1,-1],...],axis=0),axis=-1)
            # np.maximum(dlogp,0,out=dlogp)
            active_trials = active_model[range(1,self.N_f+1),...].sum(axis=-1)
            # assert np.all(dlogp>0), f'dlogp should be positive, {dlogp=}'
            
            # print(active_model[[1,2,3],...])

            reliability = active_trials / self.nSamples
            reliability_sigmoid = 1 - 1 / ( 1 + np.exp(-20*(reliability-0.3) ))
            reliability_penalty = (reliability_sigmoid * dlogp).sum(axis=0)
            # print(f"{reliability=}, {reliability_penalty=}")
            # print(f"{reliability*dlogp}")
            # reliability_penalty = penalty_factor * ((reliability>0) * (1. + (1.-reliability))).sum(axis=0)
            # reliability_penalty = penalty_factor * (1. + (1.-reliability)).sum(axis=0)
            self.timeit('reliability penalty')
        
        self.log.debug(('penalty (zeroing):',zeroing_penalty))
        self.log.debug(('penalty (centering):',centering_penalty))
        self.log.debug(('penalty (overlap):',overlap_penalty))
        self.log.debug(('penalty (reliability):',reliability_penalty))
        # self.log.debug(('penalty (activation):',activation_penalty))

        # self.log.debug(('dParams:',dParams_trial_from_total))
        lower_bound_0_penalty = np.zeros(N_in)
        for PF in params['PF']:
            for key in PF.keys():
                if np.any(PF[key]<0):
                    lower_bound_0_penalty += no_go_factor * np.sum(-PF[key],where=PF[key]<0,axis=-1)
        self.timeit('lower_bound_0 penalty')

        return zeroing_penalty + centering_penalty + overlap_penalty + reliability_penalty + lower_bound_0_penalty
        # return zeroing_penalty, centering_penalty, activation_penalty
    

    def probability_of_spike_observation(self,nu):
        ## get probability to observe N spikes (amplitude) within dwelltime for each bin in each trial
        logp = self.N*np.log(nu*self.dwelltime) - self.log_N_factorial - nu*self.dwelltime

        logp[np.logical_and(nu==0,self.N==0)] = 0
        logp[np.isnan(logp)] = -10.
        return logp


    def run_sampling(self,penalties=['overlap'],n_live=100,improvement_loops=2):

        my_prior_transform = self.set_prior_transform(vectorized=True)
        my_likelihood = self.set_logp_func(vectorized=True,penalties=penalties)

        ## setting up the sampler

        ## nested sampling parameters
        NS_parameters = {
            'min_num_live_points': n_live,
            'max_num_improvement_loops': improvement_loops,
            'max_iters': 50000,
            'cluster_num_live_points': 20,
        }

        sampler = ultranest.ReactiveNestedSampler(
            self.paramNames, 
            my_likelihood, my_prior_transform,
            wrapped_params=self.wrap,
            vectorized=True,num_bootstraps=20,
            ndraw_min=512
        )

        n_steps = 10#hbm.f * 10
        while True:
            try:
                sampler.stepsampler = PopulationSliceSampler(
                    popsize=2**4,
                    nsteps=n_steps,
                    generate_direction=generate_region_oriented_direction
                )

                sampling_result = sampler.run(
                    **NS_parameters,
                    region_class=RobustEllipsoidRegion,
                    update_interval_volume_fraction=0.01,
                    show_status=True,viz_callback=False
                )

                self.store_inference_results(sampling_result)
                break
            except:
                n_steps *= 2
                print(f'increasing step size to {n_steps=}')
            if n_steps > 100:
                break
        return sampling_result



    def store_local_parameters(self,f,posterior,key,key_results):
            
        # parameter = 
        for trial in range(self.n_trials):
            key_trial = f'{key_results}__{trial}'
            self.inference_results['parameter']['local'][key][f,trial,0] = posterior[key_trial]['mean']
            self.inference_results['parameter']['local'][key][f,trial,1:] = posterior[key_trial]['CI'][[0,-1]]

            self.inference_results['x']['local'][key][f,trial,:] = posterior[key_trial]['x']
            self.inference_results['p_x']['local'][key][f,trial,:] = posterior[key_trial]['p_x']
        pass

    def store_global_parameters(self,f,posterior,key,key_results):
        if key=='A0':
            self.inference_results['parameter']['global'][key][0] = posterior[key_results]['mean']
            self.inference_results['parameter']['global'][key][1:] = posterior[key_results]['CI'][[0,-1]]

            self.inference_results['x']['global'][key] = posterior[key_results]['x']
            self.inference_results['p_x']['global'][key] = posterior[key_results]['p_x']
        else:
            self.inference_results['parameter']['global'][key][f,0] = posterior[key_results]['mean']
            self.inference_results['parameter']['global'][key][f,1:] = posterior[key_results]['CI'][[0,-1]]

            self.inference_results['x']['global'][key][f,:] = posterior[key_results]['x']
            self.inference_results['p_x']['global'][key][f,:] = posterior[key_results]['p_x']
    
    def store_parameters(self,f,posterior,key,key_results):

        if self.priors[key_results]['n']>1:
            self.store_global_parameters(f,posterior,key,f'{key_results}__mean')
            self.store_local_parameters(f,posterior,key,key_results)
        else:
            self.store_global_parameters(f,posterior,key,key_results)
            for store_keys in ['parameter','x','p_x']:
                self.inference_results[store_keys]['local'][key] = None

    def store_inference_results(self,results):

        self.n_trials = self.nSamples
        self.inference_results = {
            'parameter': {
                ## for each parameter
                'global': {}, 	## n x N_f x 3 
                'local': {},	## n x N_f x n_trials x 3
            },
            'p_x': {
                'global': {},	## n x N_f x 100
                'local': {},	## n x N_f x n_trials x 100
            },
            'x': {
                'global': {},	## n x N_f x 100
                'local': {},	## n x N_f x n_trials x 100
            },
            'logz': 			np.zeros(2),
            'active_trials': 	np.zeros((self.N_f,self.n_trials)),
        }

        key = 'A0'
        self.inference_results['parameter']['global'][key] = np.zeros(3)
        self.inference_results['p_x']['global'][key] = np.zeros(100)
        self.inference_results['x']['global'][key] = np.zeros(100)
        
        self.inference_results['parameter']['local'][key] = np.zeros((self.n_trials,3))
        self.inference_results['p_x']['local'][key] = np.zeros((self.n_trials,100))
        self.inference_results['x']['local'][key] = np.zeros((self.n_trials,100))
    
        for key in ['theta','A','sigma']:
            self.inference_results['parameter']['global'][key] = np.zeros((self.N_f,3))
            self.inference_results['p_x']['global'][key] = np.zeros((self.N_f,100))
            self.inference_results['x']['global'][key] = np.zeros((self.N_f,100))
            
            self.inference_results['parameter']['local'][key] = np.zeros((self.N_f,self.n_trials,3))
            self.inference_results['p_x']['local'][key] = np.zeros((self.N_f,self.n_trials,100))
            self.inference_results['x']['local'][key] = np.zeros((self.N_f,self.n_trials,100))
        
        
        posterior = self.build_posterior(results)

        for key in ['A0','theta','A','sigma']:

            if key=='A0':
                ## A0 is place field-independent parameter
                self.store_parameters(0,posterior,key,key)
                pass
            else:
                ## other parameters are place-field dependent
                for f in range(self.N_f):
                    key_prior = f'PF{f+1}_{key}'
                    self.store_parameters(f,posterior,key,key_prior)
                

            self.inference_results['logz'][0] = results['logz']
            self.inference_results['logz'][1] = results['logzerr']

            N_draws = 1000
            my_logp = self.set_logp_func(penalties=['overlap','reliability'])
            
            active_model = my_logp(results['weighted_samples']['points'][-N_draws:,:],get_active_model=True)

            if self.N_f>1:
                active_model[1,active_model[-1,...]] = True
                active_model[2,active_model[-1,...]] = True
                
            for f in range(self.N_f):
                self.inference_results['active_trials'][f,...] = active_model[f+1,...].sum(axis=0)
            self.inference_results['active_trials'] /= N_draws



            # 	# then, get active_trials from my_logp. maybe check for random draws from 1000 samples of logp (in high logp region) and construct probability of activation from it
          

    def build_posterior(self,results,nsteps=101,smooth_sigma=1,use_dynesty=False):

        posterior = {}
        
        for i,key in enumerate(self.paramNames):

            if use_dynesty:
                samp = results.samples[:,i]
                weights = results.importance_weights()


            else:
                samp = results['weighted_samples']['points'][:,i]
                weights = results['weighted_samples']['weights']

            mean = (samp*weights).sum()

            # sort samples
            samples_sorted = np.sort(samp)
            idx_sorted = np.argsort(samp)

            # get corresponding weights
            sw = weights[idx_sorted]

            cumsw = np.cumsum(sw)

            quants = np.interp([0.001,0.05,0.341,0.5,0.841,0.95,0.999], cumsw, samples_sorted)
            
            low,high = quants[[0,-1]]
            x = np.linspace(low,high,nsteps)

            f = interp1d(samples_sorted,cumsw,bounds_error=False,fill_value='extrapolate')
            
            posterior[key] = {
                'CI': quants[1:-1],
                'mean': mean,
                'x': x[:-1],
                'p_x': gauss_filter(f(x[1:]) - f(x[:-1]),smooth_sigma) if smooth_sigma>0 else f(x[1:]) - f(x[:-1])
            }

        return posterior

def norm_cdf(x,mu,sigma):
    return 0.5*(1. + erf((x-mu)/(np.sqrt(2)*sigma)))






# def call_HBM(pathSession='../data/579ad/Session10',neuron=0,f=1,hierarchical=[],wrap=[],run_it=False,use_dynesty=False,logLevel=logging.ERROR,plot=False,penalties=['parameter_bias','overlap','reliability']):

#     pathBehavior = os.path.join(pathSession,'aligned_behavior.pkl')
#     pathActivity = [os.path.join(pathSession,file) for file in os.listdir(pathSession) if (file.startswith('results_CaImAn') and 'redetected' in file and not ('compare' in file))][0]
#     # pathActivity = os.path.join(pathSession,'OnACID_results.hdf5')

#     ld = load_dict_from_hdf5(pathActivity)
#     #S = gauss_filter(ld['S'][neuron,:],2)
#     S = ld['S'][neuron,:]


#     # with open(pathBehavior,'rb') as f_open:
#     #     ld = pickle.load(f_open)

#     nbin = 40
#     bin_array = np.linspace(0,nbin-1,nbin)
#     behavior = prepare_behavior_from_file(pathBehavior,nbin=nbin,f=15)
#     activity = prepare_activity(S,behavior['active'],behavior['trials'],nbin=nbin)

#     #print((S>0).sum(),activity['S'].sum())
    
#     firingstats = get_firingstats_from_trials(activity['trial_map'],behavior['trials']['dwelltime'],N_bs=1000)

#     if plot:
#         plt.figure()
#         plt.plot(activity['trial_map'].T)
#         plt.show(block=False)

#         fig = plt.figure()
#         ax = fig.add_subplot(121)   
#         ax.plot(behavior['time_raw'],activity['S'],'r',linewidth=0.3)

#         ax = fig.add_subplot(122)
#         ax.bar(bin_array,firingstats['map'],facecolor='b',width=1,alpha=0.2)
#         # ax.bar(self.para['bin_array'],fmap,facecolor='r',width=1,alpha=0.2)
#         ax.errorbar(bin_array,firingstats['map'],firingstats['CI'],ecolor='r',linestyle='',fmt='',elinewidth=0.3)
#         plt.draw()
#         plt.show(block=False)
#     # time.sleep(1)
#     # return

#     # #%matplotlib nbagg 
#     # fig,ax = plt.subplots(1,2)
#     # ax[0].plot(ld['time'],S)
#     # ax[0].scatter(behavior['time'],activity['S'],s=5,color='tab:orange')
#     # ax[1].plot(activity['trial_map'].T)
#     # plt.show(block=False)

#     hbm = HierarchicalBayesModel(
#         activity['trial_map'],
#         behavior['trials']['dwelltime'],
#         np.arange(nbin),
#         f=f,
#         hierarchical=hierarchical,
#         wrap=wrap,
#         logLevel=logLevel
#     )

#     if not run_it:
#         return hbm, None, None


    
#     if use_dynesty:
#         # my_prior_transform = lambda p_in : hbm.transform_p(p_in,vectorized=False)
#         my_prior_transform = hbm.set_prior_transform(vectorized=False)
#         my_likelihood = hbm.set_logp_func(vectorized=False)
#         print('running nested sampling')
#         # print(np.where(hbm.pTC['wrap'])[0])
#         with dypool.Pool(8,my_likelihood,my_prior_transform) as pool:
#             sampler = NestedSampler(pool.loglike,pool.prior_transform,hbm.nParams,
#                     pool=pool,
#                     nlive=100,
#                     bound='multi',
#                     # periodic=np.where(hbm.pTC['wrap'])[0],
#                     sample='rslice'
#                 )
#             sampler.run_nested()

#         sampling_result = sampler.results
#         # print(sampling_result)
#         return hbm, sampling_result, sampler
#     else:
#         # print(hbm.wrap)
#         my_prior_transform = hbm.set_prior_transform(vectorized=True)
#         my_likelihood = hbm.set_logp_func(vectorized=True,penalties=['overlap'])

#         # print(hbm.paramNames)
#         sampler = ultranest.ReactiveNestedSampler(
#             hbm.paramNames, 
#             my_likelihood, my_prior_transform,
#             wrapped_params=hbm.wrap,
#             vectorized=True,num_bootstraps=20,
#             ndraw_min=512
#         )


#         nsteps = 10#hbm.nParams
#         sampler.stepsampler = PopulationSliceSampler(
#             popsize=8,
#             nsteps=nsteps,
#             generate_direction=generate_region_oriented_direction
#         )

#         # step_matrix = np.zeros((nsteps,hbm.nParams),'bool')
#         # for key in hbm.priors:
#         #     if hbm.priors[key]['n']==1:
#         #         step_matrix[::3,hbm.priors[key]['idx']] = True
#         #     else:
#         #         step_matrix[:,hbm.priors[key]['idx']:hbm.priors[key]['idx']+hbm.priors[key]['n']] = True
#         # print(step_matrix)
#         # sampler.stepsampler = ultranest.stepsampler.SpeedVariableRegionSliceSampler(
#         #     step_matrix=step_matrix,
#         #     # nsteps=nsteps,
#         # )
#         # sampler.stepsampler = ultranest.stepsampler.SliceSampler(
#         #     nsteps=nsteps,
#         #     generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
#         #     # adaptive_nsteps=False,
#         #     # max_nsteps=400
#         # )
            
#         # sampler.stepsampler = ultranest.stepsampler.SliceSampler(
#         #     nsteps=20,
#         #     generate_direction=ultranest.stepsampler.generate_cube_oriented_direction,
#         # )
#         # sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(
#         #     nsteps=nsteps
#         # )#, adaptive_nsteps='move-distance')
#         # num_samples = hbm.nPars*100
#         num_samples = 100

#         sampling_result = sampler.run(
#             min_num_live_points=num_samples,
#             max_iters=50000,cluster_num_live_points=20,max_num_improvement_loops=3,
#             # show_status=True,viz_callback='auto')  ## ... and run it #max_ncalls=500000,(f+1)*100,
#             show_status=True,viz_callback=False)  ## ... and run it #max_ncalls=500000,(f+1)*100,
#             #region_class=ultranest.mlfriends.SimpleRegion(),

#         return hbm, sampling_result, sampler


# def analyze_results(BM,results,use_dynesty=False,correct_for_reliability=True):

#     posterior = build_posterior(BM,results)

#     if use_dynesty:
#         # from dynesty import utils as dyfunc
#         # mean,cov = dyfunc.mean_and_cov(samples,weights)
#         mean = np.full((1,BM.nParams),np.nan)
#         for i,key in enumerate(BM.paramNames):
#             mean[0,i] = posterior[key]['mean']
#     else:
#         mean = np.array(results['posterior']['mean'])[np.newaxis,:]

#     params = BM.from_p_to_params(mean)

#     my_logp = BM.set_logp_func()
#     active_model = my_logp(mean,get_active_model=True)
#     if BM.f>1:
#         active_model[1,active_model[-1,...]] = True
#         active_model[2,active_model[-1,...]] = True

#     fig = plt.figure()
#     ax = fig.add_subplot(121)

#     ax_theta = fig.add_subplot(222)

#     col = ['r','g']
    
#     if BM.f > 0:
#         theta_vals = {
#             'CI': np.zeros((BM.f,2,BM.nSamples)),
#             'mean': np.zeros((BM.f,BM.nSamples))
#         }
#         for key in BM.paramNames:
#             if key.startswith('PF'):
#                 f = int(key[2])
#                 if 'theta' in key:
#                     suffix = key.split('__')[-1]
#                     if not suffix.isdigit():
#                         continue
#                     trial = int(key.split('__')[-1])
#                     theta_vals['mean'][f-1,trial] = posterior[key]['mean']
#                     theta_vals['CI'][f-1,:,trial] = posterior[key]['CI'][[1,-2]]


#         for f,PF in enumerate(params['PF']):
#             ax.axhline(mean[0,BM.priors[f'PF{f+1}_theta__mean']['idx']],color='k',linestyle='--')

#             ax.plot(
#                 theta_vals['mean'][f,:],
#                 color='r')
            
#             ax.errorbar(
#                 np.where(active_model[f+1,0,...])[0],
#                 theta_vals['mean'][f,active_model[f+1,0,...]],
#                 np.abs(theta_vals['mean'][f,active_model[f+1,0,...]] - theta_vals['CI'][f,:,active_model[f+1,0,...]].T),
#                 color='k',marker='o')
#             ax_theta.plot(posterior[f'PF{f+1}_theta__mean']['x'],posterior[f'PF{f+1}_theta__mean']['p_x'],color=col[f])
#             for i in np.where(active_model[f+1,0,...])[0]:
#                 ax_theta.plot(np.mod(posterior[f'PF{f+1}_theta__{i}']['x'],BM.Nx),posterior[f'PF{f+1}_theta__{i}']['p_x'],color=col[f],linewidth=0.2)
#     plt.setp(ax_theta,xlim=[0,BM.Nx])
#     plt.show(block=False)

#     lw = 1
#     fig = plt.figure(figsize=(12,8))
#     for trial in range(BM.nSamples):
#         ax = fig.add_subplot(5,5,trial+1)
#         ax.plot(BM.N[0,trial,:]/BM.dwelltime[0,trial,:],'k:')
#         ax.plot(gauss_filter(BM.N[0,trial,:]/BM.dwelltime[0,trial,:],1),'k')

#         # print(params['PF'][0]['p'])
#         # isfield = params['PF'][0]['p'][0,trial] > 0.5
#         if BM.f > 0:
#             isfield = active_model[1,0,trial]
#             if BM.f>1:
#                 isfield_2 = active_model[2,0,trial]#params['PF'][1]['p'][0,trial] > 0.5
#             else:
#                 isfield_2 = False

#             if not isfield and not isfield_2:
#                 # print(BM.model_of_tuning_curve(params,None).shape)
#                 # print(BM.model_of_tuning_curve(params).shape)
#                 ax.plot(BM.x_arr[0,0,:],BM.model_of_tuning_curve(params,None)[0,trial,:],'r--',linewidth=lw)
#             if isfield and isfield_2:
#                 ax.plot(BM.x_arr[0,0,:],BM.model_of_tuning_curve(params)[0,trial,:],'r--',linewidth=lw)
            
#             if isfield and not isfield_2:
#                 ax.plot(BM.x_arr[0,0,:],BM.model_of_tuning_curve(params,0)[0,trial,:],'r--',linewidth=lw)
#             if not isfield and isfield_2:
#                 ax.plot(BM.x_arr[0,0,:],BM.model_of_tuning_curve(params,1)[0,trial,:],'r--',linewidth=lw)
#         else:
#             ax.plot(BM.x_arr[0,0,:],BM.model_of_tuning_curve(params,None)[0,trial,:],'r--',linewidth=lw)

        
#         ax.set_ylim([0,30])
    
#     plt.show(block=False)