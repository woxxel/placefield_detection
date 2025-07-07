import numpy as np
import tqdm

# from matplotlib import pyplot as plt
from pathlib import Path

from caiman.utils.utils import load_dict_from_hdf5

from ..utils import (
    prepare_behavior_from_file,
    get_firingrate,
)

from alternative_detection_methods import peak_method_batch, information_method_batch, stability_method

def detection_methods_wrapper(
    pathSession, neurons=False, nbin=100, methods=["peak"], plot=False, **kwargs
):
    pathSession = Path(pathSession)

    pathBehavior = pathSession / "aligned_behavior.pkl"
    behavior = prepare_behavior_from_file(
        pathBehavior,
        nbin=nbin,
        nbin_coarse=None,
        f=15.0,
        T=None,
        speed_gauss_sd=4,
        calculate_performance=False,
    )

    pathActivity = [
        file
        for file in pathSession.iterdir()
        if (
            file.stem.startswith("results_CaImAn")
            and not "compare" in file.stem
            and "redetected" in file.stem
        )
    ][0]
    # print(pathActivity)
    # pathActivity = os.path.join(pathSession,'OnACID_results.hdf5')
    # print(pathActivity)
    ld = load_dict_from_hdf5(pathActivity)
    # print(ld.keys())
    # S = gauss_filter(ld['S'][neuron,:],2)

    neuron_activity = ld["S"][:, behavior["active"]]
    n_neurons = neuron_activity.shape[0]

    neurons = range(n_neurons) if not neurons else list(neurons)
    # plot = plot if len(neurons) == 1 else False

    # if plot:
    #     fmap = get_firingmap(
    #         activity[neurons[0], :],
    #         behavior["binpos"],
    #         behavior["dwelltime"],
    #         nbin=nbin,
    #     )
    #     baseline = np.percentile(fmap, 50)

    #     frate, firing_threshold, significant_spikes = get_firingrate(
    #         activity[neurons[0], :], f=15.0, sd_r=-1, Ns_thr=10, prctile=20
    #     )
    #     # print(f'{frate=}')
    #     # SD = get_SD_from_half_fluctuations(fmap,baseline)

    #     fig = plt.figure()

    #     ax_activity = fig.add_subplot(221)
    #     ax_activity.plot(activity[neurons[0], :])
    #     ax_activity.axhline(firing_threshold, color="g", linestyle="--")

    #     ax = fig.add_subplot(222)
    #     ax.plot(gauss_filter(fmap, 1))
    #     # ax.axhline(baseline,color='k',linestyle='--')
    #     # ax.axhline(baseline+2*SD,color='r',linestyle='--')
    #     # ax.axhline(firing_threshold,color='r',linestyle='--')

    is_place_cell = {}
    if "peak" in methods:
        is_place_cell["peak"] = peak_method_batch(
            behavior,
            neuron_activity,
            neurons=neurons,
            nbin=nbin,
            # plot=plot,
            # ax=fig.add_subplot(234) if plot else None,
            **kwargs,
        )

    if "information" in methods:
        is_place_cell["information"] = information_method_batch(
            behavior,
            neuron_activity,
            neurons=neurons,
            nbin=nbin,
            # plot=plot,
            # ax=fig.add_subplot(235) if plot else None,
        )

    if "stability" in methods:
        is_place_cell["stability"] = stability_method(
            behavior,
            neuron_activity,
            neurons=neurons,
            nbin=nbin,
            # plot=plot,
            # ax=fig.add_subplot(236) if plot else None,
        )

    is_place_cell["rates"] = np.zeros(n_neurons)
    is_place_cell["rates_active"] = np.zeros(n_neurons)
    for neuron in tqdm.tqdm(neurons):
        is_place_cell["rates"][neuron], firing_threshold, significant_spikes = (
            get_firingrate(ld["S"][neuron, :], f=15.0, sd_r=0, Ns_thr=10, prctile=10)
        )
        is_place_cell["rates_active"][neuron], firing_threshold, significant_spikes = (
            get_firingrate(neuron_activity[neuron, :], f=15.0, sd_r=0, Ns_thr=10, prctile=10)
        )

    # if plot:
    #     # plt.tight_layout()
    #     plt.show(block=False)
    # return behavior, activity
    return is_place_cell
