import numpy as np
import scipy as sp


class HierarchicalBayesModel:

  ### possible speedup through...
  ###   parallelization of code

    def __init__(self, data, x_arr, parsNoise, f):

        self.behavior = data
        self.nbin = data.shape[0]
        self.parsNoise = parsNoise
        self.x_arr = x_arr
        self.x_max = x_arr.max()
        self.Nx = len(self.x_arr)
        self.change_model(f)

        ### steps for lookup-table (if used)
        #self.lookup_steps = 100000
        #self.set_beta_prior(5,4)

    def set_logl_func(self):

        '''
            TODO:
            instead of finding correlated trials before identification, run bayesian 
            inference on all neurons, but adjust log-likelihood:
                * don't pool trial stats, but calculate logl for each trial
                * calculate logl as probability to observe S spikes in time T (for each bin, see DM inference), 
                    thus also make sense of silent trials
                * in each trial, calculate logl for 1. no field and 2. for field
                * take placefield position, width, etc as hierarchical parameter 
                    (narrow distribution for location and baseline activity?)
                * for each trial, take the better logl as contribution to overall loglikelihood
                    (thus, avoid "activation" as parameter)
                * consider placefield only, if 10%/20%, ... of trials have activation as better logl
                * later, calculate logl for final parameter set to obtain active-trials (better logl)
            
            make sure all of this runs kinda fast!

            check, whether another hierarchy level should estimate noise-distribution parameters for overall data 
                (thus, running inference only once on complete session, with N*4 parameters)
        '''

        def get_logl(p):
            if len(p.shape)==1:
                p = p[np.newaxis,:]
            p = p[...,np.newaxis]

            mean_model = np.ones((p.shape[0],self.Nx))*p[:,0,:]
            if p.shape[1] > 1:
                for j in [-1,0,1]:   ## loop, to have periodic boundary conditions

                    mean_model += (p[:,slice(1,None,3),:]*np.exp(-(self.x_arr[np.newaxis,np.newaxis,:]-p[:,slice(3,None,3),:]+self.x_max*j)**2/(2*p[:,slice(2,None,3),:]**2))).sum(1)

            #plt.figure()
            #for i in range(p.shape[0]):
                #plt.subplot(p.shape[0],1,i+1)
                #plt.plot(self.x_arr,np.squeeze(mean_model[i,:]))
            #plt.show(block=False)

            SD_model = self.parsNoise[1] + self.parsNoise[0]*mean_model

            alpha = (mean_model/SD_model)**2
            beta = mean_model/SD_model**2

            logl = np.nansum(alpha*np.log(beta) - np.log(sp.special.gamma(alpha)) + (alpha-1)*np.log(self.behavior) - beta*self.behavior ,1)#.sum(1)
            if self.f>1:
                p_theta = p[:,slice(3,None,3)]
                dTheta = np.squeeze(np.abs(np.mod(p_theta[:,1]-p_theta[:,0]+self.nbin/2,self.nbin)-self.nbin/2))
                logl[dTheta<(self.nbin/10)] = -1e300

            return logl

        return get_logl

    ### want beta-prior for std - costly, though, so define lookup-table
    def set_beta_prior(self,a,b):
        self.lookup_beta = sp.stats.beta.ppf(np.linspace(0,1,self.lookup_steps),a,b)
        #return sp.special.gamma(a+b)/(sp.special.gamma(a)*sp.special.gamma(b))*x**(a-1)*(1-x)**(b-1)

    def transform_p(self,p):
        if p.shape[-1]>1:
            p_out = p*self.prior_stretch + self.prior_offset
            #p_out[...,2] = self.prior_stretch[2]*self.lookup_beta[(p[...,2]*self.lookup_steps).astype('int')]
        else:
            p_out = p*self.prior_stretch[0] + self.prior_offset[0]
        #print(p_out[:,slice(3,None,3)])
        return p_out

    def set_priors(self):
        self.prior_offset = np.array(np.append(0,[2,2,0]*self.f))
        prior_max = np.array(np.append(10,[100,20,self.nbin]*self.f))
        self.prior_stretch = prior_max-self.prior_offset
        #print(self.prior_stretch)
        #self.prior_stretch = np.array(np.append(1,[1,6-1,self.nbin]*self.f))

    def change_model(self,f):
        self.f = f
        self.nPars = 1+3*f
        self.TC = self.build_TC_func()
        self.pTC = {}
        self.set_priors()
        self.pTC['wrap'] = np.zeros(self.nPars).astype('bool')
        self.pTC['wrap'][slice(3,None,3)] = True

        self.transform_ct = 0

    def build_TC_func(self):        ## general function to build tuning curve model
        def TC_func(p):
            if len(p.shape)==1:
                p = p[np.newaxis,:]
            p = p[...,np.newaxis]
            TC = np.ones((p.shape[0],self.Nx))*p[:,0,:]
            if p.shape[1] > 1:
                for j in [-1,0,1]:   ## loop, to have periodic boundary conditions
                #TC += p[:,1]*np.exp(-(self.x_arr[np.newaxis,:]-p[:,3]+self.x_max*j)**2/(2*p[:,2]**2))
                    TC += (p[:,slice(1,None,3),:]*np.exp(-(self.x_arr[np.newaxis,np.newaxis,:]-p[:,slice(3,None,3),:]+self.x_max*j)**2/(2*p[:,slice(2,None,3),:]**2))).sum(1)
                    #TC += (p[:,slice(1,None,3)]*np.exp(-(self.x_arr[np.newaxis,:]-p[:,slice(3,None,3)]+self.x_max*j)**2/(2*p[:,slice(2,None,3)]**2))).sum(-1)

            return np.squeeze(TC)

        return TC_func

####------------------------ end of HBM definition ----------------------------- ####
