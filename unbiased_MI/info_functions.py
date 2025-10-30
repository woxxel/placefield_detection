import numpy as np


def compute_MI(spike_train, stimulus_trace):

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
    

    #set constant to prevent numerical issues
    epsilon = 1e-30 if spike_train.dtype == np.float32 or stimulus_trace.dtype == np.float32 else np.finfo(float).tiny
    
    #getting unique states and count occurrences
    states = np.unique(stimulus_trace)
    num_states = len(states)
    num_cells = spike_train.shape[0]
    
    #converting to integers if spike_train data has non-integer spike counts
    if np.any(spike_train % 1 != 0):
        spike_train = np.round(spike_train).astype(int)

    #getting response bins
    max_rate = np.max(spike_train)
    response_bins = np.arange(max_rate + 1)
   
    #calculating the prior distribution of the encoded variable
    p_s = np.array([np.sum(stimulus_trace == state) / len(stimulus_trace) for state in states])

    #calculating marginal probabilities for responses
    p_r = np.zeros((num_cells, len(response_bins)), order = 'F')

    for r in range(len(response_bins)):
        p_r[:, r] = np.sum(spike_train == response_bins[r], axis=1) / spike_train.shape[1]

    #calculating conditional probability
    p_r_given_s = np.zeros((num_cells, len(response_bins), num_states),order = 'F')

    for s in range(num_states):
        for r in range(len(response_bins)):
            state_indices = np.where(stimulus_trace == states[s])[0]

            if len(state_indices) > 1:
                state_spike_train = spike_train[:, state_indices]
                matches = state_spike_train == response_bins[r]
                p_r_given_s[:, r, s] = np.sum(matches, axis=1) / len(state_indices)
            else:
                matches = (spike_train[:, state_indices] == response_bins[r]).astype(int)
                p_r_given_s[:, r, s] = np.squeeze(matches)

    #calculating conditional entropy
    conditional_entropy = -np.sum(
      p_s * np.sum(p_r_given_s * np.log2(p_r_given_s + epsilon), axis=1),
      axis=1)

    #calculating response entropy
    response_entropy = -np.sum(p_r * np.log2(p_r + epsilon), axis=1)
    
    #calculating Mutual Information (MI)
    MI = response_entropy - conditional_entropy
    MI[np.isnan(MI)] = 0  #handling cases with NaN results
    
    return MI


def compute_SI(average_firing_rates, tuning_curves, stimulus_distribution):

    """ 
    Computes the naive Skaggs information index (SI) of a population of cells

    Arguments
    ----------
    average_firing_rates (np.array) :    
    tuning_curves (np.array):      
    stimulus_distribution (np.array): 

    Returns
    -------
    SI_bit_spike (np.array)  
    SI_bit_sec (np.array)   
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


def shuffle_spike_trains(spike_train, num_shuffles, shuffle_type):
    """
    Performs either a cyclic permutation or random shuffling to obtain shuffled spike trains
    
    Arguments
    ----------
    spike_train (np.array)
    num_shuffles (int): Number of shuffling repetitions
    shuffle_type (string) Either 'cyclic' or 'random' permutations
    
    Returns
    --------
    shuffled_spike_trains (tensor):  shape (N, T, K), where N is the number of neurons, T is the number of time bins, 
       and K is the number of shuffles. Each element is the spike count of a given neuron in a given time bin.
    """
    
    N,T = spike_train.shape
    shuffled_spike_trains = np.zeros((N, T, num_shuffles), order = 'F', dtype=spike_train.dtype)

    if shuffle_type == 'cyclic':
        for n in range(num_shuffles):
            shift_index = np.random.randint(T)  #randomly selecting a shift index
            shuffled_spike_trains[:, :, n] = np.roll(spike_train, shift_index, axis=0)  #performing cyclic permutation
    elif shuffle_type == 'random':
        for n in range(num_shuffles):
            random_indexes = np.argsort(np.random.rand(T))  #generating random indexes
            shuffled_spike_trains[:, :, n] = spike_train[:, random_indexes]  #random shuffling the spike train
    else:
        raise ValueError('Please choose a valid shuffling type')
    

    return shuffled_spike_trains


def compute_tuning_curves(spike_train, stimulus_trace, dt):

    """
    Computes tuning curves
    
    Arguments
    ----------
    spike_train (np.array)
    stimulus_trace (np.array)
    dt (float): Temporal bin size (in seconds)
    
    Returns
    --------
    tuning_curves (np.ndarray): firing rate map of each neuron at each position/stmuli
    stimulus_distribution (np.array): array of size S with probabilities of the each stimuli

    """

    #calculating stimulus distribution
    stimulus_values, stimulus_counts = np.unique(stimulus_trace, return_counts=True)
    stimulus_distribution = stimulus_counts / len(stimulus_trace)

    #determining the number of stimulus values, neurons, and time bins
    num_stimulus_values = len(stimulus_values)
    num_cells = spike_train.shape[0]

    #intilaizing tuning_curves
    tuning_curves = np.zeros((num_cells, num_stimulus_values), order = 'F')

    for n in range(num_stimulus_values):
        this_bin_indexes = np.where(stimulus_trace == stimulus_values[n])[0]

        if len(this_bin_indexes) > 1:
            #calculating the mean firing rate within the time bin
            tuning_curves[:, n] = np.mean(spike_train[:, this_bin_indexes], axis=1) / dt
        else:
            #using the firing rate directly if only one time bin
            tuning_curves[:, n] = spike_train[:, this_bin_indexes].flatten() / dt

    return tuning_curves, stimulus_distribution


def compute_info_versus_sample_size(spike_train, stimulus_trace, sample_sizes, dt, repetitions, info_measures):

    """
    Computes information content using multiple sample sizes
    
    Arguments
    ----------
    spike_train (np.array)
    stimulus_trace (np.array)
    sample_sizes (np.array): array of sample sizes
    dt (float): Temporal bin size (in seconds)
    repetitions (int): number of repititions for each sample size
    info_measures (np.array): binary array to indicate measures to compute (size 1*3)
    
    
    Returns
    ----------
    results (np.ndarray): information content

    """

    N,T = spike_train.shape
    sample_sizes = sample_sizes*T
    nbr_samples = len(sample_sizes)
   
    #initializing arrays to store information content
    if info_measures[0] or info_measures[1]:
        info_bit_spike_vs_sample = np.full((N, nbr_samples), np.nan, order = 'F')
        shuffle_info_bit_spike_vs_sample = np.full((N, nbr_samples), np.nan, order = 'F')
        info_bit_sec_vs_sample = np.full((N, nbr_samples), np.nan, order = 'F')
        shuffle_info_bit_sec_vs_sample = np.full((N, nbr_samples), np.nan, order = 'F')

    if info_measures[2]:
        info_mi_vs_sample = np.full((N,nbr_samples), np.nan, order = 'F')
        shuffle_info_mi_vs_sample = np.full((N, nbr_samples), np.nan, order = 'F')

    #calculating info for different sample sizes
    for n in range(nbr_samples):

        col_dim = int(np.ceil(repetitions * T / sample_sizes[n]))

        num_time_bins = int(np.floor(sample_sizes[n]))

        if info_measures[0] or info_measures[1]:
            #initializing arrays to store information content
            info_bit_spike = np.full((N, col_dim), np.nan, order = 'F')
            shuffle_info_bit_spike = np.full((N, col_dim), np.nan, order = 'F')
            info_bit_sec = np.full((N, col_dim), np.nan, order = 'F')
            shuffle_info_bit_sec = np.full((N, col_dim), np.nan, order = 'F')
        
        if info_measures[2]:
            #initializing arrays to store information content
            info_mi = np.full((N, col_dim), np.nan, order = 'F')
            shuffle_info_mi = np.full((N, col_dim), np.nan, order = 'F')
     
        for k in range(col_dim):
            #shuffling spike trains
            sample_indexes = np.argsort(np.random.rand(T))[:num_time_bins]
            shuffled_spikes =np.squeeze( shuffle_spike_trains(spike_train[:, sample_indexes], 1, 'cyclic'))
            
            if info_measures[0] or info_measures[1]:

                #computing tunung curves and calculating information content
                temp_tc, temp_states_distribution = compute_tuning_curves(spike_train[:, sample_indexes], stimulus_trace[sample_indexes], dt)
                temp_fr = np.mean(spike_train[:, sample_indexes], axis=1) / dt

                
                temp_info_bit_spike, temp_info_bit_sec = compute_SI(temp_fr, temp_tc, temp_states_distribution)

                info_bit_spike[:, k] = temp_info_bit_spike
                info_bit_sec[:, k] = temp_info_bit_sec

                temp_shuffled_tc, _ = compute_tuning_curves(shuffled_spikes, stimulus_trace[sample_indexes], dt)
                temp_shuffle_fr = np.mean(shuffled_spikes, axis=1) / dt
                temp_shuffle_info_bit_spike, temp_shuffle_info_bit_sec = compute_SI(temp_shuffle_fr, temp_shuffled_tc, temp_states_distribution)
                shuffle_info_bit_spike[:, k] = temp_shuffle_info_bit_spike
                shuffle_info_bit_sec[:, k] = temp_shuffle_info_bit_sec
             
            if info_measures[2]:
                temp_mi = compute_MI(spike_train[:, sample_indexes], stimulus_trace[sample_indexes])
                info_mi[:, k] = temp_mi
                    
                temp_mi_shuffle = compute_MI(shuffled_spikes, stimulus_trace[sample_indexes])
                shuffle_info_mi[:, k] = temp_mi_shuffle

        if info_measures[0] or info_measures[1]:
            #averaging info content across sample sizes
            info_bit_spike_vs_sample[:, n] = np.nanmean(info_bit_spike, axis=1)
            shuffle_info_bit_spike_vs_sample[:, n] = np.nanmean(shuffle_info_bit_spike, axis=1)
            info_bit_sec_vs_sample[:, n] = np.nanmean(info_bit_sec, axis=1)
            shuffle_info_bit_sec_vs_sample[:, n] = np.nanmean(shuffle_info_bit_sec, axis=1)
 
        if info_measures[2]:
            info_mi_vs_sample[:, n] = np.nanmean(info_mi, axis=1)
            shuffle_info_mi_vs_sample[:, n] = np.nanmean(shuffle_info_mi, axis=1)
                
    results = []
    if info_measures[0] or info_measures[1]:
        results.extend([info_bit_spike_vs_sample, shuffle_info_bit_spike_vs_sample, info_bit_sec_vs_sample, shuffle_info_bit_sec_vs_sample])
    if info_measures[2]:
        results.extend([info_mi_vs_sample, shuffle_info_mi_vs_sample])

    return results      


### function to shuffle spike trains according to Gansel, 2012
###
### inputs:
###         mode  - specify mode for shuffling
###               'shift'       - shift spike train by a fixed offset (default)
###                       provide values as (mode,shuffle_peaks,spike_train)
###               'dither'      - dither each spike by an independently drawn random value (max = w)
###                       provide values as (mode,shuffle_peaks,spike_times,spikes,T,ISI,w,shuffle_spikes)
###               'dithershift' - combination of 'shift' and 'dither' method
###                       provide values as (mode,shuffle_peaks,spike_times,spikes,T,ISI,w,shuffle_spikes)
###               'dsr'         - (dither-shift-reorder), same as 'dithershift' but with random reordering of consecutive ISIs < w (?)
###                       provide values as (mode,shuffle_peaks,spike_times,spikes,T,ISI,w,shuffle_spikes)
###
###         shuffle_peaks  - boolean: should assignment "spikes" to "spike_times" be shuffled?
###
###         spike_train - spike train as binary array (should be replaced by ISI & T
###
###         spike_times - frames at which spikes happen
###
###         spikes      - number of spikes happening at times "spike_times"
###
###         T     - length of the overall recording (= maximum value for new spike time)
###
###         ISI   - InterSpike Intervalls of the spike train
###
###         w     - maximum dithering (~1/(2*rate)?)
###
### ouputs:
###         new_spike_train - shuffled spike train
###
###   written by A.Schmidt, last reviewed on January, 22nd, 2020


# from numba import jit

# @jit
def shuffling(mode,shuffle_peaks,**varin):

    if mode == "shift":

        [new_spike_train, tmp] = shift_spikes(varin["spike_train"])
        if shuffle_peaks:
            spike_times = np.where(new_spike_train)[0]
            spikes = new_spike_train[spike_times]
            new_spike_train[spike_times] = spikes[
                np.random.permutation(len(spike_times))
            ]  ## shuffle spike numbers

    elif mode == "dither":

        assert (
            len(args) >= 2
        ), "You did not provide enough input. Please check the function description for further information."
        [spike_times, spikes, T, ISI, w] = get_input_dither(varin)

        new_spike_train = dither_spikes(spike_times, spikes, T, ISI, w, shuffle_peaks)

    elif mode == "dithershift":

        assert (
            len(args) >= 4
        ), "You did not provide enough input. Please check the function description for further information."
        [spike_times, spikes, T, ISI, w] = get_input_dither(varin)

        new_spike_train = dither_spikes(spike_times, spikes, T, ISI, w, shuffle_peaks)
        [new_spike_train, shift] = shift_spikes(new_spike_train)

    elif mode == "dsr":

        print("not yet implemented")
        new_spike_train = np.nan

    # plt = false;
    # if plt && strcmp(mode,'dithershift')

    # if ~exist('spike_train','var')
    # spike_train = zeros(1,T);
    # spike_train(spike_times) = spikes;
    # end
    # ISI = get_ISI(spike_train);
    # newISI = get_ISI(new_spike_train);

    # figure('position',[500 500 1200 900])
    # subplot(3,1,1)
    # plot(spike_train)
    # subplot(3,1,2)
    # plot(new_spike_train)
    # title('new spike train')

    # subplot(3,2,5)
    # hold on
    # histogram(log10(ISI),linspace(-2,2,51),'FaceColor','b')
    # histogram(log10(newISI),linspace(-2,2,51),'FaceColor','r')
    # hold off

    # waitforbuttonpress;
    # end
    return new_spike_train


def shift_spikes(spike_train,shift=None,axis=None):
  
  shift = shift if shift else np.random.randint(np.max([1,len(spike_train)]))
  new_spike_train = np.concatenate([spike_train[shift:],spike_train[:shift]])    ## shift spike train
  # new_spike_train = np.roll(spike_train,shift=shift,axis=axis)
  return new_spike_train,shift


def get_input_dither(argin):
  
  if len(argin['w']) == 1:
    spike_train = argin['spike_train']
    spike_times = np.where(spike_train)[0]
    spikes = spike_train[spike_times]
    T = len(spike_train)
    ISI = np.diff(spike_times)
    
  else:
    spike_times = argin['spike_times']
    spikes = argin['spikes']
    T = argin['T']
    ISI = argin['ISI']
    
  return spike_times,spikes,T,ISI,argin['w']


def dither_spikes(spike_times,spikes,T,ISI,w,shuffle_peaks):
  
  nspike_times = len(spike_times);
  
  dither = np.min([ISI-1,2*w])/2;
  
  r = 2*(rand(1,len(ISI)-1)-0.5);
  
  for i in range(1,len(ISI)):   ## probability of being left or right of initial spike should be equal! (otherwise, it destroys bursts!)
    print('i: %d',i)
    spike_times[i] = spike_times[i] + min(0,r[i-1])*ISI[i-1] + max(0,r[i-1])*ISI[i];
  
  spike_times = round(spike_times)
  
  if shuffle_peaks:
    print(nspike_times)
    print(nspike_times.shape)
    print('watch out: permutation only works along first dimension ... proper shape?')
    spikes = spikes[np.random.permutation(nspike_times)]
  
  new_spike_train = np.zeros(T)
  for i in range(nspike_times):
    t = spike_times[i]
    new_spike_train[t] = new_spike_train[t] + spikes[i]
  
  return new_spike_train


def get_ISI(spike_train):
  
  ## this part effectively splits up spike bursts (single event with multiple spikes to multiple events with single spikes)
  spike_times = np.where(spike_train)[0];
  idx_old = 1;
  new_spike_times = [];
  # print(np.where(spike_train>1))
  # print(np.where(spike_train>1)[0])
  for t in np.where(spike_train>1)[0]:
    idx_new = np.where(spike_times==t)[0]
    nspikes = spike_train[t]
    
    new_spike_times = np.append([new_spike_times,spike_times[idx_old:idx_new],t+np.linspace(0,1-1/nspikes,nspikes)]);
    #idx_old = idx_new+1;
  
  new_spike_times = np.append([new_spike_times,spike_times[idx_old:]]);
  return np.diff(new_spike_times);
