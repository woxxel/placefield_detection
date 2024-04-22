import numpy as np
from matplotlib import pyplot as plt
from scipy.special import factorial as sp_factorial

# from placefield_detection import *

def logl(x_arr,p):

    """
        defines the model by providing the average expected firing rate at each bin
    """
    x_max = x_arr.shape[0]

    if len(p.shape)==1:
        p = p[np.newaxis,:]
    p = p[...,np.newaxis]
    
    mean_model = np.ones((p.shape[0],x_max))*p[:,0,:]
    if p.shape[1] > 1:
        for j in [-1,0,1]:
            print(mean_model.shape)
            mean_model += (p[:,slice(1,None,3),:]*np.exp(-(x_arr[np.newaxis,np.newaxis,:]-p[:,slice(3,None,3),:]+x_max*j)**2/(2*p[:,slice(2,None,3),:]**2))).sum(1)

    return mean_model


def poisson_spikes(nu,N,T_total):
    #print("poisson:",nu,T_total)
    # print((N[:,np.newaxis]*np.log(nu[np.newaxis,:]*T_total[:,np.newaxis])).shape)
    # print((np.log(sp_factorial(N)[:,np.newaxis])).shape)
    # print((nu[np.newaxis,:]*T_total[:,np.newaxis]).shape)
    # return np.exp(N[:,np.newaxis]*np.log(nu[np.newaxis,:]*T_total[:,np.newaxis]) - np.log(sp_factorial(N)[:,np.newaxis]) - nu[np.newaxis,:]*T_total[:,np.newaxis]) 
    return np.exp(N*np.log(nu*T_total) - np.log(sp_factorial(N)) - nu*T_total) 


def test_vals(nbin,p,pd,min_trials=3):
    
    x_arr = np.arange(nbin)
    mod = logl(x_arr,np.array([[p[0],0,0,0],p]))

    p_trials = np.zeros((pd.behavior['trials']['ct'],2))

    field = np.zeros(pd.behavior['trials']['ct'],'bool')
    for t in range(pd.behavior['trials']['ct']):

        '''
            could be made vectorized...
        '''

        # get values of spikes and dwelltime for this trial
        N = pd.PC_detect.firingstats['trial_map'][t,:]
        T_total = pd.behavior['trials']['dwelltime'][t,:]

        # calculate probability to observe number of spikes (?) in given dwelltime, given the model for nofield and field 
        p_nofield_at_position = poisson_spikes(mod[0,:],N,T_total)
        p_field_at_position = poisson_spikes(mod[1,:],N,T_total)

        p_trials[t,0] = np.nansum(np.log(p_nofield_at_position))
        p_trials[t,1] = np.nansum(np.log(p_field_at_position))

    # consider trials to be place-coding, when probability is higher than nofield-model
    field = p_trials[:,1] > p_trials[:,0]

    # only consider the cell to be place coding, if at least min_trials are place-coding
    if field.sum()>min_trials:
        print('trials:',field.sum())
        p = np.max(p_trials,1).sum()
    else:
        p = p_trials[:,0].sum()
    

    # print(p_trials)
    # print(field)
    # print('probability:',p)

    # fig,ax = plt.subplots(3,1)
    # ax[0].plot(pd.PC_detect.firingstats['trial_map'].T,'gray'); 
    # ax[0].plot(pd.PC_detect.firingstats['trial_map'][np.where(field)[0],:].T,'k'); 
    
    # ax[1].plot(mod.T)

    # plt.show()

    return p