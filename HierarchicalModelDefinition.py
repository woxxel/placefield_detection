import numpy as np

class HierarchicalModel:
    """
        Defines a general class for setting up a hierarchical model for bayesian inference. Has to be inherited by a specific model class, which then further specifies the loglikelihood etc
    """

    def __init__(self,nSamples):

        self.nSamples = nSamples


    def set_priors(self,priors_init,hierarchical=[],wrap=[]):
        """
            Set the priors for the model. The priors are defined in the priors_init dictionary, which has to follow the following structure:

            priors_init = {
                **param1** : {
                    'hierarchical': {
                        'function':     **transform_function**,
                        'params': {
                            'loc':      'mean',
                            'scale':    **ref_to_2nd_key**
                        },
                    },
                    mean : {
                        'function':     **transform_function**,
                        'params':       {**params_for_function**}
                    },
                    **2nd_key** : {
                        'function':     **transform_function**,
                        'params':       {**params_for_function**}
                    }
                },
                **param2** : ...,
            }
            with **x** being placeholders for the actual values.

            All parameters appearing in "hierarchical" will be treated as hierarchical parameters. If a parameter is hierarchical, the mean and **2nd_key** parameter define the meta distribution. If a parameter is not hierarchical, only the mean parameter is used.

            TODO:
                * enable wrap function!
        """


        self.paramNames = []
        self.priors = {}
        
        ct = 0
        for param in priors_init:

            if param in hierarchical:
                
                ## add the mean and sigma parameters for the hierarchical prior
                for key in priors_init[param]['hierarchical']['params'].values():
                    
                    self.set_prior_param(priors_init, ct, param, key, hierarchical=True,meta=True)
                    ct += 1

                ## then, add the actual parameters for the hierarchical prior
                self.set_prior_param(priors_init, ct, param, hierarchical=True, meta=False)
                ct += self.nSamples
            
            else:
                ## add the parameters for the non-hierarchical prior
                self.set_prior_param(priors_init, ct, param, hierarchical=False, meta=False)
                ct += 1
        
        self.nParams = len(self.paramNames)

        self.wrap = np.zeros(self.nParams).astype('bool')
        for paramName in self.paramNames:
            try:
                key_root, key_var = paramName.split('__')
            except:
                key_root = paramName
                key_var = None
            
            if priors_init[key_root] in wrap and not key_var=='sigma':
                self.wrap[self.priors[key]['idx']:self.priors[paramName]['idx']+self.priors[paramName]['n']] = True
        
        self.wrap = np.zeros(self.nParams).astype('bool')

    

    def set_prior_param(self,priors_init,ct,param,key=None,hierarchical=False,meta=False):

        lower_hierarchy_level = (hierarchical and not meta)

        paramName = param + (f"__{key}" if key else '')

        self.priors[paramName] = {}

        var = key if key else 'mean'
    
        self.priors[paramName]['idx'] = ct
        self.priors[paramName]['n'] = self.nSamples if lower_hierarchy_level else 1
        self.priors[paramName]['meta'] = meta

        if lower_hierarchy_level:
            for i in range(self.nSamples):
                self.paramNames.append(f'{param}__{i}')

            # get indexes of hierarchical parameters for quick access later on
            self.priors[paramName]['idx_mean'] = self.priors[f"{param}__{priors_init[param]['hierarchical']['params']['loc']}"]['idx']
            self.priors[paramName]['idx_sigma'] = self.priors[f"{param}__{priors_init[param]['hierarchical']['params']['scale']}"]['idx']


            self.priors[paramName]['transform'] = \
                lambda x,params,fun=priors_init[param]['hierarchical']['function']: fun(x,**params)
        else:
            self.paramNames.append(paramName)
        
            self.priors[paramName]['transform'] = \
                lambda x,params=priors_init[param][var]['params'],fun=priors_init[param][var]['function']: fun(x,**params)
            
    

    def set_prior_transform(self,vectorized=True):
        '''
            sets the prior transform function for the model

            only takes as input the mode, which can be either of
            - 'vectorized': vectorized prior transform function
            - 'scalar': scalar prior transform function
            - 'tensor': tensor prior transform function
        '''

        def prior_transform(p_in):

            """
                The actual prior transform function, which transforms the random variables from the unit hypercube to the actual priors
            """
        
            if len(p_in.shape)==1:
                p_in = p_in[np.newaxis,...]
            p_out = np.zeros_like(p_in)
            
            for key in self.priors:
                
                if self.priors[key]['n']==1:
                    p_out[:,self.priors[key]['idx']] = self.priors[key]['transform'](p_in[:,self.priors[key]['idx']])

                else:
                    params = {
                        'loc':      p_out[:,self.priors[key]['idx_mean'],np.newaxis],
                        'scale':    p_out[:,self.priors[key]['idx_sigma'],np.newaxis],
                    }

                    p_out[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']] = self.priors[key]['transform'](p_in[:,self.priors[key]['idx']:self.priors[key]['idx']+self.priors[key]['n']],params=params)
            
            if vectorized:
                return p_out
            else:
                return p_out[0,:]
            
        return prior_transform