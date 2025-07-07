""" contains various useful program snippets for neuron analysis:

  get_nFolder   get number of folders in path
  pathcat       attach strings to create proper paths
  _hsm          half sampling mode to obtain baseline


"""

import scipy as sp

import numpy as np
import matplotlib.pyplot as plt

from .process_activity import get_spiking_data


def prepare_quantiles(C, bh_active, f=15.0, qtl_steps=4):

    activity = {}
    activity["C"] = C[:, bh_active]

    ### calculate firing rate
    activity["spikes"], activity["firing_rate"], _ = get_spiking_data(
        activity["C"], f=f, Ns_thr=1, prctile=20
    )

    # ## obtain quantized firing rate for MI calculation
    if activity["firing_rate"] > 0:
        sigma = 5

        activity["qtl"] = sp.ndimage.gaussian_filter(
            activity["C"].astype("float") * f, sigma
        )
        # activity['qtl'] = activity['qtl'][self.behavior['active']]
        qtls = np.quantile(
            activity["qtl"][activity["qtl"] > 0], np.linspace(0, 1, qtl_steps + 1)
        )

        for i in range(activity["qtl"].shape[0]):
            activity["qtl"][i] = np.count_nonzero(
                activity["qtl"][i, :, np.newaxis] >= qtls[np.newaxis, 1:-1], axis=1
            )

    return activity["qtl"]


# def get_spikeNr(data):

#     if np.count_nonzero(data) == 0:
#         return 0, np.NaN, np.NaN
#     else:
#         md = calculate_hsm(data, True)
#         #  Find the mode

#         # only consider values under the mode to determine the noise standard deviation
#         ff1 = data - md
#         ff1 = -ff1 * (ff1 < 0)

#         # compute 25 percentile
#         ff1.sort()
#         ff1[ff1 == 0] = np.NaN
#         Ns = round((ff1 > 0).sum() * 0.5)  # .astype('int')

#         # approximate standard deviation as iqr/1.349
#         iqr_h = ff1[-Ns]
#         sd_r = 2 * iqr_h / 1.349
#         data_thr = md + 2 * sd_r
#         spikeNr = np.floor(data / data_thr)  # .sum()
#         return spikeNr, md, sd_r


# from scipy._lib._array_api import array_namespace, size as xp_size


def _circfuncs_common(samples, high, low):
    # xp = array_namespace(samples) if xp is None else xp

    # if xp.isdtype(samples.dtype, "integral"):
    #     dtype = xp.asarray(1.0).dtype  # get default float type
    #     samples = xp.asarray(samples, dtype=dtype)

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    sin_samp = np.sin((samples - low) * 2.0 * np.pi / (high - low))
    cos_samp = np.cos((samples - low) * 2.0 * np.pi / (high - low))

    return samples, sin_samp, cos_samp


def circmean(
    samples, weights=1.0, high=2 * np.pi, low=0, axis=None, nan_policy="propagate"
):

    # xp = array_namespace(samples)
    # Needed for non-NumPy arrays to get appropriate NaN result
    # Apparently atan2(0, 0) is 0, even though it is mathematically undefined
    # if (samples. == 0:
    #     return xp.mean(samples, axis=axis)
    samples, sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_sum = np.sum(sin_samp * weights, axis=axis)
    cos_sum = np.sum(cos_samp * weights, axis=axis)
    res = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)

    res = res[()] if res.ndim == 0 else res
    return res * (high - low) / 2.0 / np.pi + low


from scipy.optimize import linear_sum_assignment


def thresholded_linear_sum_assignment(cost, threshold):
    """
    function to perform imbalanced, incomplete linear_sum_assignment (from scipy.optimize)
    with thresholded values (upper bound for cost function)
    """

    n_reference, n_target = cost.shape

    matched_reference, matched_target = linear_sum_assignment(cost)

    cost_matched = cost[matched_reference, matched_target]
    idx_well_matched = np.where(cost_matched < threshold)[0]

    if len(idx_well_matched):
        matched_reference = matched_reference[idx_well_matched]
        matched_target = matched_target[idx_well_matched]
    else:
        matched_reference = np.array([], "int")
        matched_target = np.array([], "int")

    non_matched_reference = np.setdiff1d(np.arange(n_reference), matched_reference)
    non_matched_target = np.setdiff1d(np.arange(n_target), matched_target)

    return matched_reference, matched_target, non_matched_reference, non_matched_target
