import numpy as np
import random, tqdm

from ..utils import (
    get_firingmap,
    get_dwelltime,
)


def stability_method(
    behavior, neuron_activity, neurons=None, plot=False, ax=None, f=15.0
):
    """
    runs a placefield detection using the stability method, as suggested in Grijseels et al, 2021
    (only works as batch processing)

    [1.] calculate separate firing maps from first and second half of recording session
    [2.] calculate correlation between two firing maps
    [3.] draw random cells from same dataset and calculate firing maps of second half
    [4.] calculate correlation between first fmap of reference vs fmap of random (create distribution)
    [5.] consider to be place field/place cell if correlation in top 5 percentile of distribution
    """

    ## some preprocessing
    n_neurons, T = neuron_activity.shape
    neurons = range(n_neurons) if neurons is None else list(neurons)

    nbin = behavior['dwelltime'].shape[0]
    
    half_fmaps = np.zeros((n_neurons, 2, nbin))
    for neuron in range(n_neurons):
        # first half
        dwelltime = get_dwelltime(behavior["binpos"][: T // 2], nbin, f)
        half_fmaps[neuron, 0, :] = get_firingmap(
            neuron_activity[neuron, : T // 2],
            behavior["binpos"][: T // 2],
            dwelltime,
            nbin=nbin,
        )

        # second half
        dwelltime = get_dwelltime(behavior["binpos"][T // 2 :], nbin, f)
        half_fmaps[neuron, 1, :] = get_firingmap(
            neuron_activity[neuron, T // 2 :],
            behavior["binpos"][T // 2 :],
            dwelltime,
            nbin=nbin,
        )

    is_place_cell = np.zeros(neuron_activity.shape[0], "bool")
    for neuron in tqdm.tqdm(neurons):
        corr_original = np.corrcoef(half_fmaps[neuron, 0, :], half_fmaps[neuron, 1, :])[
            0, 1
        ]

        n_shuffles = 100
        shuffled_corr = np.zeros(n_shuffles)
        for L, n_shuffle in enumerate(
            random.sample(range(neuron_activity.shape[0]), n_shuffles)
        ):

            shuffled_corr[L] = np.corrcoef(
                half_fmaps[neuron, 0, :], half_fmaps[n_shuffle, 1, :]
            )[0, 1]

        if plot:
            ax.hist(shuffled_corr, bins=21, alpha=0.6)
            ax.axvline(corr_original, color="g", linestyle="--")
            ax.axvline(np.nanpercentile(shuffled_corr, 95), color="r", linestyle="--")
            ax.set_title("stability")
            ax.spines[["top", "right"]].set_visible(False)

        is_place_cell[neuron] = corr_original > np.nanpercentile(shuffled_corr, 95)

    return is_place_cell