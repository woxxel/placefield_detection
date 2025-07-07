import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy import stats

from functools import partial
from concurrent.futures import ProcessPoolExecutor

from placefield_detection.utils import (
    get_firingmap,
    shift_spikes,
)

def SI(fmap, f_mean):
    """
        Calculate the spatial information (SI) from the firing map.
        as suggested in Skaggs, 1993
    """
    return np.nansum(fmap * np.log2(fmap / f_mean))


def information_method_single(
    behavior,
    neuron_activity,
    n_shuffles=1000,
    shuffle_trials=False,
    plot=False,
    ax=None,
):
    """
        runs a placefield detection using the information method, as suggested in Grijseels et al 2021 / XY
        [1.] calculate spatial information from SI, according to Skaggs (1993) formula
        [2.] shuffle $1000$ times and create distribution of SI
        [3.] consider place cell, of SI in top 5 percentile
    """
    
    if plot and ax is None:
        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_subplot(111)

    nbin = behavior['dwelltime'].shape[0]
    fmap = get_firingmap(
        neuron_activity,
        behavior["position"],
        behavior["dwelltime"],
        nbin=nbin,
    )

    f_mean = neuron_activity.mean()

    SI_original = SI(fmap, f_mean)

    shuffled_SI = np.zeros(n_shuffles)
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
        shuffled_SI[L] = SI(shuffled_fmap, f_mean)

    if plot:
        ax.hist(shuffled_SI, bins=21, alpha=0.6)
        ax.axvline(SI_original, color="g", linestyle="--",label="original SI")
        ax.axvline(np.percentile(shuffled_SI, 95), color="r", linestyle="--",label="95th percentile")
        ax.set_title("Information method (SI)")
        ax.spines[["top", "right"]].set_visible(False)

    p_value = 1 - stats.percentileofscore(shuffled_SI, SI_original, kind="rank")/100.
    # is_place_cell = SI_original > np.percentile(shuffled_SI, 95)
    is_place_cell = p_value < 0.05
    # percentile = percentileofscore(shuffled_SI, SI_original)
    # print(f"{percentile=}")
    return {"is_place_cell":is_place_cell, "p_value":p_value}


def information_method_batch(
    behavior,
    neuron_activity,
    neurons=None,
    n_shuffles=1000,
    shuffle_trials=False
):
    """
        run batch processing of information method for place field detection
    """

    n_neurons = neuron_activity.shape[0]
    neurons = range(n_neurons) if (neurons is None) else list(neurons)
    is_place_cell = np.zeros(n_neurons, "bool")
    p_value = np.full(n_neurons, np.NaN)

    process_single = partial(
        information_method_single,
        behavior,
        n_shuffles=n_shuffles,
        shuffle_trials=shuffle_trials,
        plot=False,
        ax=None,
    )

    with ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(process_single, [neuron_activity[neuron, :] for neuron in neurons]),total=len(neurons)))

    for idx, neuron in enumerate(neurons):
        results["is_place_cell"][neuron] = results[idx]["is_place_cell"]
        results["p_value"][neuron] = results[idx]["p_value"]

    return results