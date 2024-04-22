''' contains various useful program snippets for neuron analysis:

  get_nFolder   get number of folders in path
  pathcat       attach strings to create proper paths
  _hsm          half sampling mode to obtain baseline


'''

import pickle
from scipy import signal
from scipy.ndimage import binary_opening, gaussian_filter1d as gauss_filter
import numpy as np
import matplotlib.pyplot as plt

from .utils import gauss_smooth, get_firingmap, get_firingrate

def prepare_activity(S,bh_active,bh_trials,nbin=100,f=15.,calc_MI=False,qtl_steps=4):

    # key_S = 'spikes' if self.para['modes']['activity']=='spikes' else 'S'

    activity = {}
    activity['S'] = S[bh_active]

    ### calculate firing rate
    activity['firing_rate'], _, activity['spikes'] = get_firingrate(activity['S'],f=f,sd_r=0,Ns_thr=1,prctile=20)

    # activity['s'] = activity[key_S]
    
    # ## obtain quantized firing rate for MI calculation
    # if calc_MI == 'MI' and firing_rate>0:
    #     sigma = 5
    #     activity['qtl'] = sp.ndimage.gaussian_filter(activity['S'].astype('float')*f,sigma)
    #     # activity['qtl'] = activity['qtl'][self.behavior['active']]
    #     qtls = np.quantile(activity['qtl'][activity['qtl']>0],np.linspace(0,1,qtl_steps+1))
    #     activity['qtl'] = np.count_nonzero(activity['qtl'][:,np.newaxis]>=qtls[np.newaxis,1:-1],1)
    

    ## obtain trial-specific activity
    activity['trials'] = {}
    activity['trial_map'] = np.zeros((bh_trials['ct'],nbin))    ## preallocate

    
    for t in range(bh_trials['ct']):
        activity['trials'][t] = {}
        activity['trials'][t]['S'] = activity['S'][bh_trials['start'][t]:bh_trials['start'][t+1]]#gauss_smooth(active['S'][self.behavior['trials']['frame'][t]:self.behavior['trials']['frame'][t+1]]*self.para['f'],self.para['f']);    ## should be quartiles?!
        
        # ## prepare quantiles, if MI is to be calculated
        # if calc_MI == 'MI' and firing_rate>0:
        #     activity['trials'][t]['qtl'] = activity['qtl'][bh_trials['start'][t]:bh_trials['start'][t+1]];    ## should be quartiles?!

        # if self.para['modes']['activity'] == 'spikes':
        #     activity['trials'][t]['spike_times'] = np.where(activity['trials'][t]['s'])
        #     activity['trials'][t]['spikes'] = activity['trials'][t]['s'][activity['trials'][t]['spike_times']]
        #     activity['trials'][t]['ISI'] = np.diff(activity['trials'][t]['spike_times'])

        activity['trials'][t]['rate'] = activity['trials'][t]['S'].sum()/(bh_trials['nFrames'][t]/f)

        if activity['trials'][t]['rate'] > 0:
            activity['trial_map'][t,:] = get_firingmap(
                activity['trials'][t]['S'],
                bh_trials['binpos'][t],
                bh_trials['dwelltime'][t,:],
                nbin
            )#/activity['trials'][t]['rate']
    return activity

    
def prepare_behavior(time_in,position_in,rw_loc_in=None,nbin=100,nbin_coarse=None,f=15.,T=None,
                               speed_gauss_sd=4,calculate_performance=False):
    '''
        loads behavior from specified path
        Requires file to contain a dictionary with values for each frame, aligned to imaging data:
            * time      - time in seconds
            * position  - mouse position
            * active    - boolean array defining active frames (included in analysis)

    '''
    
    if T is None:
        T = time_in.shape[0]
    
    time = time_in
    position = position_in
    

    binpos,environment_length = calculate_binpos(position,nbin)

    velocity = gauss_filter(np.maximum(0,np.diff(position,prepend=position[0])),speed_gauss_sd)* nbin/environment_length * f

    data = get_trials_and_active(binpos,velocity,nbin,f,T)

    ### preparing data for active periods, only
    data['nFrames'] = np.count_nonzero(data['active'])
    
    data['binpos'] = binpos[data['active']]
    data['time'] = time[data['active']]
    data['velocity'] = velocity[data['active']]
    
    data['dwelltime'] = get_dwelltime(data['binpos'],nbin,f)
    
    if nbin_coarse:
        data['binpos_coarse'],_ = calculate_binpos(position,nbin_coarse)
        data['binpos_coarse'] = data['binpos_coarse'][data['active']]
        
        data['dwelltime_coarse'] = get_dwelltime(data['binpos_coarse'],nbin_coarse,f) 

    
    if calculate_performance:
        rw_pos = rw_loc_in*nbin
        try:
            data['performance'] = get_performance(binpos,velocity,time,rw_pos,0,nbin,f)
        except:
            pass

    # if plot_bool:
    #     plt.figure(dpi=300)
    #     plt.plot(data['time'],data['position'],'r.',markersize=1,markeredgecolor='none')
    #     plt.plot(data['time'][data['active']],data['position'][data['active']],'k.',markersize=2,markeredgecolor='none')
    #     plt.show(block=False)
    return data


def get_trials_and_active(position,velocity,
                     nbin=100,f=15.,T=None,
                     speed_thr=2.,binary_morph_width=5):
    
    data = {}
    
    inactive = binary_opening(velocity <= speed_thr,np.ones(binary_morph_width))
    data['active'] = ~inactive


    ## define trials
    data['trials'] = get_trial_data(position[data['active']],nbin,f)

    if data['trials']['start'][-1] == data['active'].sum():
        active_start = np.where(data['active'])[0][data['trials']['start'][0]]
        active_end = np.where(data['active'])[0][-1]+1
    else:
        trials_start = np.where(data['active'])[0][data['trials']['start']]
        active_start = trials_start[0]
        active_end = trials_start[-1]
    
    data['active'][:active_start] = False
    data['active'][active_end:] = False

    data['trials']['start'] -= data['trials']['start'][0]

    return data


def get_performance(binpos,velocity,time,rw_pos,rw_delay,nbin,f,plt_trials=False,plt_bool=False):

    trials = get_trial_data(binpos,nbin,f)

    ## get behavior performance
    rw_end = rw_pos+nbin/5
    rw_delay=0

    range_approach = [-2,4]      ## in secs
    ra = range_approach[1]-range_approach[0]
    
    ra_frames = int(f*ra)
    ra_idxs = np.arange(int(range_approach[0]*f),int(range_approach[1]*f))
    ra_arr = np.linspace(range_approach[0],range_approach[1],ra_frames)

    vel_max = np.ceil(np.max(velocity)/10)*10
    vel_arr = np.linspace(vel_max/50,vel_max,51)
    
    hist = gauss_smooth(np.histogram(velocity,vel_arr)[0],2,mode='nearest')

    vel_run_idx = signal.find_peaks(hist,distance=10,prominence=vel_max*0.3)[0][-1]
    # vel_run = vel_arr[vel_run_idx]
    vel_min_idx = signal.find_peaks(-hist,distance=5)[0]
    vel_min_idx = vel_min_idx[vel_min_idx<vel_run_idx][-1]
    vel_thr = vel_arr[vel_min_idx]

    # performance = {}
    performance = {}
    performance['RW_reception'] = np.zeros(trials['ct'],'bool')
    performance['RW_frame'] = np.zeros(trials['ct'],'int')

    performance['slowDown'] = np.zeros(trials['ct'],'bool')
    performance['frame_slowDown'] = np.zeros(trials['ct'],'int')
    performance['pos_slowDown'] = np.full(trials['ct'],np.NaN)
    performance['t_slowDown_beforeRW'] = np.full(trials['ct'],np.NaN)
    
    performance['RW_approach_time'] = np.zeros((trials['ct'],int(ra*f)))
    performance['RW_approach_space'] = np.full((trials['ct'],nbin),np.NaN)
    
    if plt_trials:
        _,axx = plt.subplots(trials['ct']//5+1,5)
        _,ax = plt.subplots(trials['ct']//5+1,5)

    for t in range(trials['ct']):
        pos_trial = binpos[trials['start'][t]:trials['start'][t+1]].astype('int')
        vel_trial = velocity[trials['start'][t]:trials['start'][t+1]]
        time_trial = time[trials['start'][t]:trials['start'][t+1]]
        for j,p in enumerate(range(nbin)):
            performance['RW_approach_space'][t,j] = vel_trial[pos_trial==p].mean()
        
        idx_enterRW = np.where(pos_trial>rw_pos)[0][0]         ## find, where first frame within rw position is
        idx_RW_reception = int(idx_enterRW + rw_delay*f)

        performance['RW_approach_time'][t,:] = vel_trial[ra_idxs+idx_RW_reception]

        if plt_trials:
            ax[t//5,t%5].plot(performance['RW_approach_space'][t,:])
            axx[t//5,t%5].plot(vel_trial,'k-')

            ax[t//5,t%5].axhline(vel_thr,color='k',linestyle='--')
            ax[t//5,t%5].plot(pos_trial[idx_enterRW],vel_trial[idx_enterRW],'rx')
            ax[t//5,t%5].plot(pos_trial[idx_RW_reception],vel_trial[idx_RW_reception],'ro')

            axx[t//5,t%5].axhline(vel_thr,color='k',linestyle='--')
            axx[t//5,t%5].plot(idx_RW_reception,vel_trial[idx_RW_reception],'ro')


        if pos_trial[idx_RW_reception]<rw_end:
            performance['RW_frame'][t] = trials['start'][t] + idx_RW_reception
            performance['RW_reception'][t] = True
            idx_trough_tmp = signal.find_peaks(-vel_trial,prominence=2,height=-vel_thr,distance=f)[0]


            # print('trough_tmp: ',idx_trough_tmp)
            idx_trough_tmp = idx_trough_tmp[idx_trough_tmp>idx_enterRW]

            if plt_trials:
                for i in idx_trough_tmp:
                    ax[t//5,t%5].plot(pos_trial[i],vel_trial[i],'go')

            if len(idx_trough_tmp)>0:
                # idx_trough = idx_enterRW + idx_trough_tmp[0]
                idx_trough = idx_trough_tmp[0]
                ### slowing down should occur before this - defined by drop below threshold velocity
                slow_down = np.where(
                    (vel_trial[:idx_trough]>vel_thr) & \
                    (pos_trial[:idx_trough]<=rw_end) & \
                    (pos_trial[:idx_trough]>5)
                )[0][-1]

                if plt_trials:
                    ax[t//5,t%5].plot(pos_trial[slow_down],vel_thr,'go',markersize=5)
                    ax[t//5,t%5].plot(pos_trial[idx_trough],vel_thr,'ro',markersize=5)

                    axx[t//5,t%5].plot(slow_down,vel_trial[slow_down],'go',markersize=5)
                    axx[t//5,t%5].plot(idx_trough,vel_trial[idx_trough],'ro',markersize=5)

                # print('velocity @ trial ',t,':  ',vel_trial[slow_down+1:int(slow_down+1+f)].mean(),vel_thr)
                if vel_trial[slow_down+1:int(slow_down+1+f)].mean() < vel_thr :#vel_trial[slow_down+1]<vel_thr:
                    performance['slowDown'][t] = True
                    performance['frame_slowDown'][t] = trials['start'][t] + slow_down
                    performance['pos_slowDown'][t] = pos_trial[slow_down]
                    performance['t_slowDown_beforeRW'][t] = time_trial[idx_RW_reception] - time_trial[slow_down]
    if plt_trials:
        plt.show(block=False)
        
    if plt_bool:

        # fig,ax = plt.subplots(2,1,figsize=(8,5))
        # ax[0].plot(time,binpos)
        # ax[1].plot(time,velocity)
        # # for s in trial_start:
        # #     ax[0].axvline(time[s],color='r')
        # #     ax[1].axvline(time[s],color='r')
        # plt.show(block=False)
        
        # fig,ax = plt.subplots(2,1,figsize=(8,5))
        # # print(data['bi'])
        # ax[0].plot(data['binpos'])
        # ax[1].plot(data['velocity'])
        # for s in data['trials']['start']:
        #     ax[0].axvline(s,color='r')
        #     ax[1].axvline(s,color='r')
        # plt.show(block=False)
        

        plt.figure()

        plt.subplot(221)
        plt.plot(ra_arr,performance['RW_approach_time'].T,color=[0.5,0.5,0.5],alpha=0.5)
        plt.plot(ra_arr,performance['RW_approach_time'].mean(0),color='k')
        plt.plot(-performance['t_slowDown_beforeRW'][performance['slowDown']],velocity[performance['frame_slowDown'][performance['slowDown'][:]]],'rx')
        plt.axhline(vel_thr,color='k',linestyle='--',linewidth=0.5)
        # plt.xlim(range_approach)

        plt.subplot(222)
        plt.plot(np.linspace(0,nbin-1,nbin),performance['RW_approach_space'].T,color=[0.5,0.5,0.5],alpha=0.5)
        plt.plot(np.linspace(0,nbin-1,nbin),np.nanmean(performance['RW_approach_space'],0),color='k')
        plt.plot(performance['pos_slowDown'][performance['slowDown']],velocity[performance['frame_slowDown'][performance['slowDown'][:]]],'rx')
        plt.plot([0,nbin],[vel_thr,vel_thr],'k--',linewidth=0.5)

        ax = plt.subplot(223)
        ax.hist(velocity,np.linspace(vel_max/10,vel_max,51))
        ax.plot(np.linspace(vel_max/10,vel_max,50),hist)
        ax.plot([vel_thr,vel_thr],[0,ax.get_ylim()[-1]],'k--')

        plt.show(block=False)
        
    return performance


def get_trial_data(position,nbin,f,partial_threshold=0.6):

    trials = {}
    ## define start points
    trials['start'] = np.hstack([0,np.where(np.diff(position)<(-nbin/2))[0] + 1,position.shape[0]])

    if not (position[0] < nbin*(1-partial_threshold)):
        # print('remove partial first trial @',trials['start'][0])
        # data['active'][:max(0,trial_start[0])] = False
        trials['start'] = trials['start'][1:]

    if not (position[-1] >= nbin*partial_threshold):
        # print('remove partial last trial @',trials['start'][-1])
        trials['start'] = trials['start'][:-1]
        # data['active'][trial_start[-1]:] = False

    # trials['start_t'] = data['time'][trials['start'][:-1]]
    trials['ct'] = len(trials['start']) - 1


    ## getting trial-specific behavior data
    trials['dwelltime'] = np.zeros((trials['ct'],nbin))
    trials['nFrames'] = np.zeros(trials['ct'],'int')
    trials['binpos'] = {}

    for t in range(trials['ct']):
        trials['binpos'][t] = position[trials['start'][t]:trials['start'][t+1]]
        trials['dwelltime'][t,:] = get_dwelltime(trials['binpos'][t],nbin,f)
        trials['nFrames'][t] = len(trials['binpos'][t])
    return trials


def get_dwelltime(position,nbin,f):
    
    bin_array = np.arange(nbin+1)-0.5

    ## define dwelltimes
    dwelltime = np.histogram(
            position,
            bin_array
        )[0]/f
    
    return dwelltime


def calculate_binpos(position,nbin):

    ## get range of values
    min_val,max_val = np.nanpercentile(position,(0.1,99.9))
    environment_length = max_val - min_val

    binpos = np.minimum((position - min_val) / environment_length * nbin,nbin-1).astype('int')
    return binpos, environment_length



def get_spikeNr(data):

    if np.count_nonzero(data)==0:
        return 0,np.NaN,np.NaN
    else:
        md = calculate_hsm(data,True);       #  Find the mode

        # only consider values under the mode to determine the noise standard deviation
        ff1 = data - md;
        ff1 = -ff1 * (ff1 < 0);

        # compute 25 percentile
        ff1.sort()
        ff1[ff1==0] = np.NaN
        Ns = round((ff1>0).sum() * .5).astype('int')

        # approximate standard deviation as iqr/1.349
        iqr_h = ff1[-Ns];
        sd_r = 2 * iqr_h / 1.349;
        data_thr = md+2*sd_r;
        spikeNr = np.floor(data/data_thr).sum();
        return spikeNr,md,sd_r