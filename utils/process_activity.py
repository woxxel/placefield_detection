import numpy as np
from matplotlib import pyplot as plt

from .utils_various import estimate_stats_from_one_sided_process

def prepare_activity(
    neuron_activity,
    behavior,
    f=15.0,
    only_active=True,
    calc_MI=False,
    qtl_steps=4,
    use_thresholded=False,
):

    # key_S = 'spikes' if self.para['modes']['activity']=='spikes' else "spikes"

    activity = {}
    # activity["spikes"] = S[bh_active]

    ### calculate firing rate
    activity["spikes"], activity["firing_rate"], _ = get_spiking_data(
        neuron_activity, f=f, Ns_thr=1, prctile=20
    )
    # print(activity["spikes"].shape)
    activity["spikes_original"] = activity["spikes"].copy()
    if only_active:
        activity["spikes"] = activity["spikes"][behavior["active"]]

    activity["map_rates"] = get_firingmap(
        activity["spikes"],
        behavior["position"],
        behavior["dwelltime"],
        nbin=behavior["nbin"],
    )

    # activity["spikes"] = activity[key_S]
    # S_active = S.copy()
    # ## obtain quantized firing rate for MI calculation
    # if calc_MI == 'MI' and firing_rate>0:
    #     sigma = 5
    #     activity['qtl'] = sp.ndimage.gaussian_filter(activity["spikes"].astype('float')*f,sigma)
    #     # activity['qtl'] = activity['qtl'][self.behavior['active']]
    #     qtls = np.quantile(activity['qtl'][activity['qtl']>0],np.linspace(0,1,qtl_steps+1))
    #     activity['qtl'] = np.count_nonzero(activity['qtl'][:,np.newaxis]>=qtls[np.newaxis,1:-1],1)

    ## obtain trial-specific activity
    activity["trials"] = {}
    activity["map_trial_rates"] = np.zeros(
        (behavior["trials"]["ct"], behavior["nbin"])
    )  ## preallocate
    activity["map_trial_spikes"] = np.zeros(
        (behavior["trials"]["ct"], behavior["nbin"])
    )  ## preallocate

    for t in range(behavior["trials"]["ct"]):
        activity["trials"][t] = {}
        activity["trials"][t]["spikes"] = activity["spikes"][
            behavior["trials"]["start"][t] : behavior["trials"]["start"][t + 1]
        ]  
        
        # ## prepare quantiles, if MI is to be calculated
        # if calc_MI == 'MI' and firing_rate>0:
        #     activity['trials'][t]['qtl'] = activity['qtl'][behavior["trials"]['start'][t]:behavior["trials"]['start'][t+1]];    ## should be quartiles?!

        activity["trials"][t]["rate"] = activity["trials"][t]["spikes"].sum() / (
            behavior["trials"]["nFrames"][t] / f
        )

        if activity["trials"][t]["rate"] > 0:
            activity["map_trial_rates"][t, :] = get_firingmap(
                activity["trials"][t]["spikes"],
                behavior["trials"]["position"][t],
                behavior["trials"]["dwelltime"][t, :],
                behavior["nbin"],
            )
            activity["map_trial_spikes"][t, :] = get_firingmap(
                activity["trials"][t]["spikes"],
                behavior["trials"]["position"][t],
                nbin=behavior["nbin"],
            )
    return activity


def get_spiking_data(neuron_activity, f=15, Ns_thr=1, prctile=20):
    """
    calculates the firing rate from a an array of spike probabilities ('S' from CaImAn)
    by thresholding data according to multiples sd_r of estimated variance

    Ns_thr:
      - minimum number of non-zero entries in S


    returns
        - firing rate (in spikes/sec, depending on provided framerate f)
        - calculated spiking threshold
        - array with number of spikes per frame

    """

    max_burst_rate = 200.0  # spikes per second
    max_spikes_per_frame = max_burst_rate // f

    neuron_activity[neuron_activity < neuron_activity.max() * 10 ** (-3)] = 0
    Ns = (neuron_activity > 0).sum()
    if Ns < Ns_thr:
        return 0, np.NaN, np.zeros_like(neuron_activity)

    # trace = neuron_activity[neuron_activity > 0]

    baseline, _ = estimate_stats_from_one_sided_process(
        neuron_activity,
        baseline_mode="percentile",
        prctile=prctile,
        only_nonzero_entries=True,
    )

    activity = np.clip(neuron_activity / baseline, 0, max_spikes_per_frame)
    activity[np.logical_and(activity > 0.1, activity < 1)] = 1.0
    activity = np.floor(activity)

    # significant_events, threshold, sd_r = (
    #     obtain_significant_events_from_one_sided_process(
    #         trace, sd_r=sd_r, baseline_mode="hsm", sd_mode="iqr", prctile=prctile
    #     )
    # )

    # number of spikes in each bin is the value multiple above calculated threshold
    # activity = np.floor(significant_events)
    # activity = np.ceil(S / firing_threshold_adapt)
    # N_spikes = activity.sum()

    return (
        activity,
        np.mean(activity) * f,
        baseline,
    )  # S > firing_threshold_adapt#



def get_firingmap(spikes, binpos, dwelltime=None, nbin=None):

    if not nbin:
        nbin = np.max(binpos) + 1

    ### calculates the firing map
    spike_times = np.where(spikes)
    spikes = spikes[spike_times]
    binpos = binpos[spike_times]  # .astype('int')

    firingmap = np.zeros(nbin)
    for p, s in zip(binpos, spikes):  # range(len(binpos)):
        firingmap[p] = firingmap[p] + s

    if not (dwelltime is None):
        firingmap = firingmap / dwelltime
        firingmap[dwelltime == 0] = np.NaN

    return firingmap


def plot_activity(
    activity,
    behavior,
    f=15.0,
    only_active=True,
    use_thresholded=False,
):
    """
    plots the activity of a neuron
    """
    # if activity is None:
    #     activity = prepare_activity(
    #         activity, behavior, f=f, only_active=only_active, use_thresholded=use_thresholded
    #     )

    # fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
    fig = plt.figure(figsize=(8,4))

    ax_spikes = fig.add_subplot([0.1,0.5,0.6,0.35])
    ax_spikes.plot(behavior["time_original"], activity["spikes_original"],'k-',label="all spikes")#,linewidth=1.)
    ax_spikes.plot(behavior["time_original"][~behavior["active"]], activity["spikes_original"][~behavior["active"]], 'r-',label="spikes during rest")#,linewidth=1.)
    ax_spikes.legend()

    spike_times = np.where(activity["spikes_original"] > 0)[0]
    spike_times_active = np.where(np.logical_and(activity["spikes_original"]>0,behavior["active"]))[0]
    plt.setp(ax_spikes,ylabel="# spikes", xticklabels=[])

    ax_position = fig.add_subplot([0.1,0.1,0.6,0.35],sharex=ax_spikes)
    ax_position.plot(behavior["time_original"], behavior["position_original"], 'k.',markersize=1,markeredgecolor='none')
    ax_position.plot(behavior["time_original"][spike_times], behavior["position_original"][spike_times], 'r.', markersize=1)
    ax_position.plot(behavior["time_original"][spike_times_active], behavior["position_original"][spike_times_active], 'k.', markersize=2)
    plt.setp(ax_position,ylim=[0,behavior["nbin"]],xlabel="Time (s)",ylabel="Position (bin)")
    
    ax_fmap = fig.add_subplot([0.75,0.1,0.2,0.35])
    ax_fmap.barh(np.arange(0,behavior["nbin"]), activity["map_rates"], height=1.,color='k')
    
    plt.setp(ax_fmap,ylim=[0,behavior["nbin"]],xlabel="rate Ca-transients [Hz]")
    # axx.set_

    for axx in [ax_spikes, ax_position, ax_fmap]:
        # axx.set_xlim(behavior["time_original"][0], behavior["time_original"][-1])
        axx.spines[["top", "right"]].set_visible(False)
        # axx.set_ylabel("Activity")
    plt.show()
    
    # return fig, axes