import numpy as np
import random, tqdm

from pathlib import Path
from caiman.utils.utils import load_dict_from_hdf5
from .utils import get_firingmap, prepare_behavior_from_file, shift_spikes, get_dwelltime, prepare_activity, get_firingrate
from scipy.ndimage import gaussian_filter1d as gauss_filter

from matplotlib import pyplot as plt


def peak_method_wrapper(pathSession,neurons=False,nbin=100,methods=['peak'],plot=False,**kwargs):
    pathSession = Path(pathSession)

    pathBehavior = pathSession / 'aligned_behavior.pkl'
    behavior = prepare_behavior_from_file(pathBehavior,nbin=100,nbin_coarse=None,f=15.,T=None,speed_gauss_sd=4,calculate_performance=False)

    pathActivity = [file for file in pathSession.iterdir() if (file.stem.startswith('results_CaImAn') and not 'compare' in file.stem and 'redetected' in file.stem)][0]
    # print(pathActivity)
    # pathActivity = os.path.join(pathSession,'OnACID_results.hdf5')
    # print(pathActivity)
    ld = load_dict_from_hdf5(pathActivity)
    # print(ld.keys())
    #S = gauss_filter(ld['S'][neuron,:],2)


    activity = ld['S'][:,behavior['active']]

    neurons = range(activity.shape[0]) if not neurons else list(neurons)
    plot = plot if len(neurons) == 1 else False
    
    if plot:
        fmap = get_firingmap(activity[neurons[0],:],behavior['binpos'],behavior['dwelltime'],nbin=nbin)
        baseline = np.percentile(fmap,50)

        frate, firing_threshold, significant_spikes = get_firingrate(activity[neurons[0],:],f=15.,sd_r=-1,Ns_thr=10,prctile=20)
        # print(f'{frate=}')
        # SD = get_SD_from_half_fluctuations(fmap,baseline)

        fig = plt.figure()

        ax_activity = fig.add_subplot(221)
        ax_activity.plot(activity[neurons[0],:])
        ax_activity.axhline(firing_threshold,color='g',linestyle='--')

        ax = fig.add_subplot(222)
        ax.plot(gauss_filter(fmap,1))
        # ax.axhline(baseline,color='k',linestyle='--')
        # ax.axhline(baseline+2*SD,color='r',linestyle='--')
        # ax.axhline(firing_threshold,color='r',linestyle='--')


    is_place_cell = {}
    if 'peak' in methods:
        is_place_cell['peak'] = peak_method(behavior,ld['S'],neurons=neurons,nbin=nbin,plot=plot,ax=fig.add_subplot(234) if plot else None,**kwargs)
    
    if 'information' in methods:
        is_place_cell['information'] = information_method(behavior,activity,neurons=neurons,nbin=nbin,plot=plot,ax=fig.add_subplot(235) if plot else None)
    
    if 'stability' in methods:
        is_place_cell['stability'] = stability_method(behavior,activity,neurons=neurons,nbin=nbin,plot=plot,ax=fig.add_subplot(236) if plot else None)
    
    is_place_cell['rates'] = np.zeros(activity.shape[0])
    is_place_cell['rates_active'] = np.zeros(activity.shape[0])
    for neuron in tqdm.tqdm(neurons):
        is_place_cell['rates'][neuron], firing_threshold, significant_spikes = get_firingrate(ld['S'][neuron,:],f=15.,sd_r=0,Ns_thr=10,prctile=10)
        is_place_cell['rates_active'][neuron], firing_threshold, significant_spikes = get_firingrate(activity[neuron,:],f=15.,sd_r=0,Ns_thr=10,prctile=10)

    if plot:
        # plt.tight_layout()
        plt.show(block=False)
    # return behavior, activity
    return is_place_cell



def peak_method(behavior,S,neurons=0,nbin=100,n_shuffles=1000,n_bootstraps=1,jackknife=False,shuffle_trials=False,plot=False,ax=None):
    '''
        runs a placefield detection using the peak method, as suggested in XY

        n_bootstraps=1 corresponds to no bootstrapping at all, but applying algorithm to whole data
    '''
    ### before that, write wrapper function to load behavior and activity data to then call this function

    
    ## [1.] calculate firing rate map

    # fmaps = np.zeros((S.shape[0],nbin))
    # for neuron in neurons:
    #     fmaps[neuron] = get_firingmap(S[neuron,:],behavior['binpos'],behavior['dwelltime'],nbin=nbin)
    
    ## [2.] obtain peak heights and position (guess from thresholding)

    is_place_cell = np.zeros(S.shape[0],'bool')
    frates = np.zeros(S.shape[0])
    f=15.
    if jackknife:
        n_bootstraps = behavior['trials']['ct']

    if plot:
        fig = plt.figure()

    for neuron in tqdm.tqdm(neurons):

        # print(behavior['trials']['start'])
        activity = prepare_activity(S[neuron,:],behavior['active'],behavior['trials'],nbin=nbin)

        # print(f'{frate=}')
        
        # neuron_activity = activity[neuron,:]
        # fmap = get_firingmap(activity['S'],behavior['binpos'],behavior['dwelltime'],nbin=nbin)
        # PF_height = np.max(fmap[neuron])

        # if PF_height < baseline+4*SD:
        #     print('maximum not high enough')
        #     return
        # plot those onto the firingmap plot!

        ## [3.] shuffle data N times (500 or more) and apply step 1 and 2

        PF_height = np.zeros(n_bootstraps)
        shuffled_PF_height = np.zeros((n_bootstraps,n_shuffles))

        is_place_cell_bootstraps = np.zeros(n_bootstraps,'bool')

        for B in range(n_bootstraps):
            if jackknife:
                bootstrapped_samples = list(range(behavior['trials']['ct']))
                bootstrapped_samples.remove(B)
            elif n_bootstraps>1:
                bootstrapped_samples = random.sample(range(behavior['trials']['ct']),behavior['trials']['ct']//2)
            else:
                bootstrapped_samples = list(range(behavior['trials']['ct']))
            
            # print(bootstrapped_samples)

            neuron_activity_bootstrapped = np.hstack([activity['trials'][t]['S'] for t in bootstrapped_samples])

            binpos_bootstrapped = np.hstack([behavior['trials']['binpos'][t] for t in bootstrapped_samples])

            dwelltime_bootstrapped = get_dwelltime(binpos_bootstrapped,nbin,f)
            
            fmap_bootstrapped = get_firingmap(neuron_activity_bootstrapped,binpos_bootstrapped,dwelltime_bootstrapped,nbin=nbin)

            PF_height[B] = np.max(fmap_bootstrapped)    
            for L in range(n_shuffles):
                
                shuffled_neuron_activity,shift = shift_spikes(neuron_activity_bootstrapped,break_points=behavior['trials']['start'] if shuffle_trials else None)

                shuffled_fmap = get_firingmap(shuffled_neuron_activity,binpos_bootstrapped,dwelltime_bootstrapped,nbin=nbin)

                shuffled_PF_height[B,L] = np.max(shuffled_fmap)

            is_place_cell_bootstraps[B] = PF_height[B] > np.percentile(shuffled_PF_height[B,:],95)

            if plot:
                axx = fig.add_subplot(5,int(n_bootstraps//5)+1,B+1)
                axx.hist(shuffled_PF_height[B,:],bins=21,alpha=0.6)
                axx.axvline(PF_height[B],color='g',linestyle='--')
                axx.axvline(np.percentile(shuffled_PF_height[B,:],95),color='r',linestyle='--')
                axx.set_title('peak')
                axx.spines[['top','right']].set_visible(False)
            # is_place_cell[neuron] = PF_height > np.percentile(shuffled_PF_height,95)
        
        is_place_cell[neuron] = is_place_cell_bootstraps.sum()/n_bootstraps > 0.5
        if plot:
            for B in range(n_bootstraps):
                ax.hist(shuffled_PF_height[B,:],bins=21,alpha=0.6)
                ax.axvline(PF_height[B],color='g',linestyle='--')
                ax.axvline(np.percentile(shuffled_PF_height[B,:],95),color='r',linestyle='--')
                ax.set_title('peak')
                ax.spines[['top','right']].set_visible(False)

    return is_place_cell

    ## [4.] generate distribution of peak heights
    # return PF_height > np.percentile(shuffled_PF_height,95)


    ## [5.] consider peak to be place field, if peak is in top 5 percentile of distribution

def information_method(behavior,activity,fmaps=None,neurons=None,nbin=100,shuffle_trials=False,plot=False,ax=None):
    '''
        runs a placefield detection using the information method, as suggested in XY
    
        [1.] calculate spatial information from $SI = \sum_{i=1}^{N} f_i \log{\frac{f_1}{f}}$ with $f_i$ fluorescence in bin $i$ and $f$ average fluorescence.
        [2.] shuffle $1000$ times and create distribution of SI
        [3.] consider place cell, of SI in top 5 percentile
    '''

    fmaps = np.zeros((activity.shape[0],nbin))
    for neuron in neurons:
        fmaps[neuron] = get_firingmap(activity[neuron,:],behavior['binpos'],behavior['dwelltime'],nbin=nbin)
    
    ## [2.] obtain peak heights and position (guess from thresholding)

    def SI(fmap,f_mean):
        return np.nansum(fmap * np.log2(fmap/f_mean))
    
    is_place_cell = np.zeros(activity.shape[0],'bool')

    n_shuffles = 1000

    for neuron in tqdm.tqdm(neurons):
        ## [1.] calculate spatial information
        neuron_activity = activity[neuron,:]
        
        f_mean = neuron_activity.mean()

        SI_original = SI(fmaps[neuron,:],f_mean)

        shuffled_SI = np.zeros(n_shuffles)
        for L in range(n_shuffles):
            
            shuffled_neuron_activity,shift = shift_spikes(neuron_activity,break_points=behavior['trials']['start'] if shuffle_trials else None)

            shuffled_fmap = get_firingmap(shuffled_neuron_activity,behavior['binpos'],behavior['dwelltime'],nbin=nbin)
            shuffled_SI[L] = SI(shuffled_fmap,f_mean)
        
        if plot:
            ax.hist(shuffled_SI,bins=21,alpha=0.6)
            ax.axvline(SI_original,color='g',linestyle='--')
            ax.axvline(np.percentile(shuffled_SI,95),color='r',linestyle='--')
            ax.set_title('SI')
            ax.spines[['top','right']].set_visible(False)

        is_place_cell[neuron] = SI_original > np.percentile(shuffled_SI,95)
    
    return is_place_cell



def stability_method(behavior,activity,neurons=None,nbin=100,plot=False,ax=None,f=15.):
    '''
        runs a placefield detection using the stability method, as suggested in XY

        [1.] calculate separate firing maps from first and second half of recording session
        [2.] calculate correlation between two firing maps
        [3.] draw random cells from same dataset and calculate firing maps of second half
        [4.] calculate correlation between first fmap of reference vs fmap of random (create distribution)
        [5.] consider to be place field/place cell if correlation in top 5 percentile of distribution
    '''

    ## some preprocessing
    half_fmaps = np.zeros((activity.shape[0],2,nbin))
    N,T = activity.shape
    for neuron in range(N):
        # first half
        dwelltime = get_dwelltime(behavior['binpos'][:T//2],nbin,f)
        half_fmaps[neuron,0,:] = get_firingmap(activity[neuron,:T//2],behavior['binpos'][:T//2],dwelltime,nbin=nbin)
        
        # second half
        dwelltime = get_dwelltime(behavior['binpos'][T//2:],nbin,f)
        half_fmaps[neuron,1,:] = get_firingmap(activity[neuron,T//2:],behavior['binpos'][T//2:],dwelltime,nbin=nbin)
    

    is_place_cell = np.zeros(activity.shape[0],'bool')
    for neuron in tqdm.tqdm(neurons):
        corr_original = np.corrcoef(half_fmaps[neuron,0,:],half_fmaps[neuron,1,:])[0,1]

        n_shuffles = 100
        shuffled_corr = np.zeros(n_shuffles)
        for L,n_shuffle in enumerate(random.sample(range(activity.shape[0]),n_shuffles)):

            shuffled_corr[L] = np.corrcoef(half_fmaps[neuron,0,:],half_fmaps[n_shuffle,1,:])[0,1]
        
        if plot:
            ax.hist(shuffled_corr,bins=21,alpha=0.6)
            ax.axvline(corr_original,color='g',linestyle='--')
            ax.axvline(np.nanpercentile(shuffled_corr,95),color='r',linestyle='--')
            ax.set_title('stability')
            ax.spines[['top','right']].set_visible(False)

        is_place_cell[neuron] = corr_original > np.nanpercentile(shuffled_corr,95)
    
    return is_place_cell




def get_SD_from_half_fluctuations(data,threshold):

    ff1 = data - threshold
    ff1 = -ff1 * (ff1 < 0)

    # compute 25 percentile
    ff1.sort()
    ff1[ff1==0] = np.NaN
    Ns = round((ff1>0).sum() * .5)#.astype('int')

    # approximate standard deviation as iqr/1.349
    iqr_h = ff1[-Ns]
    sd_r = 2 * iqr_h / 1.349
    return sd_r