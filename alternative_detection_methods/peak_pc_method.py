import numpy as np
import tqdm
from scipy import stats

from matplotlib import pyplot as plt
from functools import partial
import concurrent.futures


from placefield_detection.utils import (
    get_firingmap,
    shift_spikes,
)

def peak_method_single(
    behavior,
    neuron_activity,
    n_shuffles=1000,
    shuffle_trials=False,
    plot=False,
    ax=None,
):
    """runs a placefield detection using the peak method, as suggested in Grijseels et al 2021 / Fournier et al 2020
    [1.] calculate firing rate map
    [2.] obtain maximum peak height
    [3.] shuffle data N times (500 or more) and apply step 1 and 2
    [4.] generate distribution of peak heights
    [5.] consider peak to be place field, if peak is in top 5 percentile of distribution
    """
    nbin = behavior['dwelltime'].shape[0]
    fmap = get_firingmap(
        neuron_activity,
        behavior["position"],
        behavior["dwelltime"],
        nbin=nbin,
    )
    PF_height = np.max(fmap)

    ## generate distribution of peak heights
    shuffled_PF_height = np.zeros(n_shuffles)
    for L in range(n_shuffles):

        shuffled_neuron_activity, shift = shift_spikes(
            neuron_activity,
            break_points=(behavior["trials"]["start"] if shuffle_trials else None),
        )

        shuffled_fmap = get_firingmap(
            shuffled_neuron_activity,
            behavior["position"],
            behavior["dwelltime"],
            nbin=nbin,
        )
        shuffled_PF_height[L] = np.max(shuffled_fmap)

    ## consider peak to be place field, if peak is in top 5 percentile of distribution
    # is_place_cell = PF_height > np.percentile(shuffled_PF_height, 95)
    p_value = 1 - stats.percentileofscore(shuffled_PF_height, PF_height, kind="rank")/100.
    is_place_cell = p_value < 0.05

    if plot:
        if ax is None:
            fig = plt.figure(figsize=(4,2))
            ax = fig.add_subplot(111)

        ax.hist(shuffled_PF_height, bins=21, alpha=0.6)
        ax.axvline(PF_height, color="g", linestyle="--",label="original peak")
        ax.axvline(
            np.percentile(shuffled_PF_height, 95),
            color="r",
            linestyle="--",
            label="95th percentile",
        )
        ax.set_title("peak method")
        ax.legend()

        ax.spines[["top", "right"]].set_visible(False)

    return {"is_place_cell":is_place_cell, "p_value":p_value}


def peak_method_batch(
    behavior,
    neuron_activity,
    neurons=None,
    n_shuffles=1000,
    shuffle_trials=False,
):
    """
        batch processing of the peak method for place field detection
    """

    n_neurons = neuron_activity.shape[0]
    neurons = range(n_neurons) if not neurons else list(neurons)
    results = {
        "is_place_cell": np.zeros(n_neurons, "bool"),
        "p_value": np.full(n_neurons, np.NaN),
    }
    # is_place_cell = np.zeros(n_neurons, "bool")    
    # p_value = np.full(n_neurons, np.NaN)

    process_single = partial(
        peak_method_single,
        behavior,
        n_shuffles=n_shuffles,
        shuffle_trials=shuffle_trials,
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(
            executor.map(process_single, neuron_activity),
            total=len(neurons)
        ))

    for idx, neuron in enumerate(neurons):
        results["is_place_cell"][neuron] = results[idx]["is_place_cell"]
        results["p_value"][neuron] = results[idx]["p_value"]

    return results