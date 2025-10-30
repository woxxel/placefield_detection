import time
import numpy as np

def compute_SI(average_firing_rates, tuning_curves, stimulus_distribution):
    """
        This function performs the naive calculation of the Skaggs information index (SI) for a population of cells.

        Inputs:
            1. average_firing_rates - Vector of size N with the average firing rates of each cell
            2. tuning_curves - Matrix of size NxS with the firing rate map of each
            3. stimulus_distribution - Vector of size S with the probabilities of the different stimuli (stimuli)

        Outputs:
            1. SI_bit_spike - Vector of size N with the naive Skaggs information in bit/spike of each cell
            2. SI_bit_sec - Vector of size N with the naive Skaggs information in bit/sec of each cell
    """

    #set constant to prevent numerical issues
    epsilon = 1e-30 if any(arr.dtype == np.float32 for arr in (average_firing_rates, tuning_curves, stimulus_distribution)) else np.finfo(float).tiny

    #normalizing tuning curves
    norm_tuning_curves = tuning_curves / (average_firing_rates[:, np.newaxis] + epsilon)

    #calculating SI in bits per spike
    SI_bit_spike = np.nansum(stimulus_distribution * norm_tuning_curves * np.log2(norm_tuning_curves + epsilon), axis=1)
    SI_bit_spike[average_firing_rates == 0] = np.nan
    
    #calculating SI in bits per sec
    SI_bit_sec = SI_bit_spike * average_firing_rates
    SI_bit_sec[np.isnan(SI_bit_spike)] = 0

    return SI_bit_spike, SI_bit_sec


def compute_MI(spike_train, stimulus_trace, epsilon=1e-30,print_time=False):

    """ 
    Computes the naive mutual information of a population of cells

    Arguments
    ----------
    spike_train (np.ndarray) :    spike count of a neuron (N) in a time bin (T) - size (T * N)  
    stimulus_trace (np.array):   state/position at T    
    
    Returns
    -------
    MI (np.array):       naive mutual information for each neuron (N)
    """

    # set constant to prevent numerical issues
    # epsilon = 1e-30 if spike_train.dtype == np.float32 or stimulus_trace.dtype == np.float32 else np.finfo(float).tiny

    # getting unique states and count occurrences
    states = np.unique(stimulus_trace)
    # print(states)
    num_states = len(states)
    num_cells, T = spike_train.shape

    # converting to integers if spike_train data has non-integer spike counts
    if print_time:
        t_spikes = time.time()

    if spike_train.dtype != int:
        # if np.any(spike_train % 1 != 0):
        # spikes = np.copy(spike_train)
        spike_train[spike_train == 0] = np.nan
        thr = np.nanmedian(spike_train, axis=1)
        spike_train = np.clip(np.ceil(spike_train / thr[:, None]),a_min=0,a_max=30)

    # getting response bins
    max_rate = np.nanmax(spike_train)
    response_bins = np.arange(max_rate + 1)

    if print_time:
        print('t spikes:', (time.time() - t_spikes)*1000)
        t_p_s = time.time()

    # calculating the prior distribution of the encoded variable (dwelltime)
    p_s = np.array([np.sum(stimulus_trace == state) for state in states]) / len(stimulus_trace)
    if print_time:
        print('time dwelltime:', (time.time() - t_p_s)*1000)
        t_p_r = time.time()

    # calculating marginal probabilities for responses
    p_r = np.zeros((num_cells, len(response_bins)))#, order = 'F')
    for r,response_bin in enumerate(response_bins):
        p_r[:, r] = np.sum(spike_train == response_bin, axis=1)
    p_r /= T

    if print_time:
        print('time p_r:', (time.time() - t_p_r)*1000)
        t_p_joint = time.time()

    # calculating conditional probability
    p_r_given_s = np.zeros((num_cells, len(response_bins), num_states))#,order = 'F')

    for s, state in enumerate(states): # iterate through locations
        state_indices = stimulus_trace == state    # get activity at location states[s]
        state_spike_train = spike_train[:, state_indices]
        nS = state_indices.sum()

        for r,response_bin in enumerate(response_bins): # iterate through spike-#
            p_r_given_s[:, r, s] = np.sum(state_spike_train == response_bin, axis=1) / nS

    if print_time:
        print('time p_joint:', (time.time() - t_p_joint)*1000)
        t_post = time.time()
    # calculating conditional entropy
    conditional_entropy = -np.sum(
      p_s * np.sum(p_r_given_s * np.log2(p_r_given_s + epsilon), axis=1),
      axis=1)

    # calculating response entropy
    response_entropy = -np.sum(p_r * np.log2(p_r + epsilon), axis=1)

    # calculating Mutual Information (MI)
    MI = response_entropy - conditional_entropy
    MI[np.isnan(MI)] = 0  #handling cases with NaN results
    if print_time:
        print('time post:', (time.time() - t_post)*1000)

    return MI


def get_MI(p_joint,p_x,p_f):

    ### - joint distribution
    ### - behavior distribution
    ### - firing rate distribution
    ### - all normalized, such that sum(p) = 1

    p_tot = p_joint * np.log2(p_joint/(p_x[:,np.newaxis]*p_f[np.newaxis,:]))
    return np.nansum(p_tot)


def get_info_value(activity,dwelltime,mode='MI'):

        if mode == 'MI':

            p_joint = np.zeros((self.para['nbin_coarse'],self.para['qtl_steps']))
            for q in range(self.para['qtl_steps']):
                for (x,ct) in Counter(self.behavior['binpos_coarse'][activity==q]).items():
                    p_joint[x,q] = ct
            p_joint = p_joint/p_joint.sum();    ## normalize

            return get_MI(p_joint,dwelltime/dwelltime.sum(),np.full(self.para['qtl_steps'],1/self.para['qtl_steps']))

        # elif mode == 'Isec':
        #     fmap = get_firingmap(activity,self.behavior['binpos_coarse'],dwelltime,nbin=self.para['nbin_coarse'])
        #     Isec_arr = dwelltime/dwelltime.sum()*(fmap/np.nanmean(fmap))*np.log2(fmap/np.nanmean(fmap))

        #     #return np.nansum(Isec_arr[-self.para['nbin']//2:])
        #     return np.nansum(Isec_arr)


def get_p_joint(activity):

    ### need as input:
    ### - activity (quantiled or something)
    ### - behavior trace
    p_joint = np.zeros((self.para['nbin_coarse'],self.para['qtl_steps']))

    for q in range(self.para['qtl_steps']):
        for (x,ct) in Counter(self.behavior['binpos_coarse'][activity==q]).items():
            p_joint[x,q] = ct
    p_joint = p_joint/p_joint.sum();    ## normalize
    return p_joint
