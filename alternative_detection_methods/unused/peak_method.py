import numpy as np
import random

from matplotlib import pyplot as plt

from ...utils import (
    get_firingmap,
    shift_spikes,
    get_dwelltime,
)

def peak_method(
    behavior,
    neuron_activity,
    nbin=100,
    n_shuffles=1000,
    n_bootstraps=1,
    jackknife=False,
    shuffle_trials=False,
    plot=False,
    ax=None,
):
    """ 
        peak method, including bootstrapping / jackknifing for more rigorous testing

        however, did not really make any difference in the end, so not used anymore
    """
    # print(behavior['trials']['start'])
    # activity = prepare_activity(
    #     neuron_activity, behavior["active"], behavior["trials"], nbin=nbin
    # )

    ## [3.] shuffle data N times (500 or more) and apply step 1 and 2
    if plot and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    PF_height = np.zeros(n_bootstraps)
    shuffled_PF_height = np.zeros((n_bootstraps, n_shuffles))

    is_place_cell_bootstraps = np.zeros(n_bootstraps, "bool")

    for B in range(n_bootstraps):
        if jackknife:
            bootstrapped_samples = list(range(behavior["trials"]["ct"]))
            bootstrapped_samples.remove(B)
        elif n_bootstraps > 1:
            bootstrapped_samples = random.sample(
                range(behavior["trials"]["ct"]), behavior["trials"]["ct"] // 2
            )
        else:
            bootstrapped_samples = list(range(behavior["trials"]["ct"]))

        # print(bootstrapped_samples)

        # neuron_activity_bootstrapped = np.hstack(
        #     [activity["trials"][t]["S"] for t in bootstrapped_samples]
        # )
        neuron_activity_bootstrapped = np.hstack(
            [
                neuron_activity[
                    behavior["trials"]["start"][t] : behavior["trials"]["start"][t + 1]
                ]
                for t in bootstrapped_samples
            ]
        )

        binpos_bootstrapped = np.hstack(
            [behavior["trials"]["binpos"][t] for t in bootstrapped_samples]
        )

        dwelltime_bootstrapped = get_dwelltime(binpos_bootstrapped, nbin, f=15.0)

        fmap_bootstrapped = get_firingmap(
            neuron_activity_bootstrapped,
            binpos_bootstrapped,
            dwelltime_bootstrapped,
            nbin=nbin,
        )

        PF_height[B] = np.max(fmap_bootstrapped)
        for L in range(n_shuffles):

            shuffled_neuron_activity, shift = shift_spikes(
                neuron_activity_bootstrapped,
                break_points=(behavior["trials"]["start"] if shuffle_trials else None),
            )

            shuffled_fmap = get_firingmap(
                shuffled_neuron_activity,
                binpos_bootstrapped,
                dwelltime_bootstrapped,
                nbin=nbin,
            )

            shuffled_PF_height[B, L] = np.max(shuffled_fmap)

        is_place_cell_bootstraps[B] = PF_height[B] > np.percentile(
            shuffled_PF_height[B, :], 95
        )
        # percentile = percentileofscore(shuffled_PF_height[B, :], PF_height[B])
        # print(f"{percentile=}")

        if plot and n_bootstraps > 1:
            axx = fig.add_subplot(5, int(n_bootstraps // 5) + 1, B + 1)
            axx.hist(shuffled_PF_height[B, :], bins=21, alpha=0.6)
            axx.axvline(PF_height[B], color="g", linestyle="--")
            axx.axvline(
                np.percentile(shuffled_PF_height[B, :], 95),
                color="r",
                linestyle="--",
            )
            axx.set_title("peak")
            axx.spines[["top", "right"]].set_visible(False)
        # is_place_cell[neuron] = PF_height > np.percentile(shuffled_PF_height,95)

    is_place_cell = is_place_cell_bootstraps.sum() / n_bootstraps > 0.5
    if plot:
        for B in range(n_bootstraps):
            ax.hist(shuffled_PF_height[B, :], bins=21, alpha=0.6)
            ax.axvline(PF_height[B], color="g", linestyle="--")
            ax.axvline(
                np.percentile(shuffled_PF_height[B, :], 95),
                color="r",
                linestyle="--",
            )
            ax.set_title("peak")
            ax.spines[["top", "right"]].set_visible(False)

    ## [4.] generate distribution of peak heights
    # return PF_height > np.percentile(shuffled_PF_height,95)

    ## [5.] consider peak to be place field, if peak is in top 5 percentile of distribution
    return is_place_cell