import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks,peak_widths

from matplotlib import pyplot as plt

from placefield_detection.utils import prepare_activity, estimate_stats_from_one_sided_process


def thresholding_method_single(
    behavior,
    neuron_activity,
    threshold_factor=4,
    sigma=4,
    plot=False
):
    """
        method to detect place fields, inspired by XY
    """
    ### calculate (and smooth) firing rate map

    activity = prepare_activity(
        neuron_activity,
        behavior,
        f=15.0,
        only_active=False,
    )
    # fmap = get_firingmap(
    #     neuron_activity,
    #     behavior["position"],
    #     behavior["dwelltime"],
    #     nbin=behavior["nbin"],
    # )
    
    place_fields = {}
    if np.sum(activity["map_rates"]>0) == 0:
        return place_fields

    # print(neuron_activity[neuron_activity > 0])
    # print(fmap)

    smooth_map = gaussian_filter1d(activity["map_rates"], sigma=sigma, mode = 'wrap')
    baseline, sd = estimate_stats_from_one_sided_process(smooth_map, baseline_mode="percentile", prctile=50, only_nonzero_entries=False)
    
    ### thresholding to detect place fields
    threshold = baseline + (threshold_factor*sd)
    field_locations, _ = find_peaks(smooth_map, height=threshold, distance = 10, width = 2)

    field_locations = field_locations.tolist()

    PF_amplitude = []
    place_field_width = []
    centroids = []

    if field_locations:
        PF_amplitude = smooth_map[field_locations].tolist()

        # calculating centroids for each place field
        for i, loc in enumerate(field_locations):

            """
                some remarks on namras method:
                    - for centroid calculation in this way, map should be shifted for each peak, as map with multiple peaks will have a peak somewhere (on border?)
                        -> fixed that
                    - place field width here is calculated on smoothed map, thus information is largely overwritten by gaussian smoothing
                        -> reduced
                    
            """
            
            # shift map, so that the peak is in the center
            shift_amount = len(smooth_map) // 2 - loc
            shifted_map = np.roll(smooth_map, shift_amount)

            # print("loc:", loc, "shift_amount:", shift_amount)
            width, _, left_ip, right_ip = peak_widths(shifted_map, [(loc+shift_amount)%len(smooth_map)], rel_height=0.6)
            # print(widths)
            place_field_width.append(width[0])

            left_idx = int(np.floor(left_ip))
            right_idx = int(np.ceil(right_ip))
            field_indices = np.arange(left_idx, right_idx + 1)
            field_activities = shifted_map[field_indices]

            #computing center of mass
            centroid = np.sum(field_indices * field_activities) / np.sum(field_activities)
            centroid = (centroid - shift_amount)% len(smooth_map)
            centroids.append(centroid)
            # centroids.append(circmean(shifted_map, high=nbin, low=0))
        
        place_fields = {
            'n_modes': len(centroids),
            'baseline': baseline,
            'amplitude': PF_amplitude,
            'location': centroids,
            'width': place_field_width,
        } 

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(activity["map_rates"], label='firing rate map')
        ax.plot(smooth_map, label='smoothed firing rate map')
        ax.axhline(baseline, color='k', linestyle='--', label='baseline')
        ax.axhline(threshold, color='r', linestyle='--', label='threshold')

        if len(field_locations) > 0:
            ax.plot(field_locations, smooth_map[field_locations], 'ro', label='place field locations')
            ax.plot(centroids, PF_amplitude, 'go', label='centroids')
            for i, loc in enumerate(field_locations):
                ax.hlines(
                    y=smooth_map[loc] - 0.6 * (smooth_map[loc] - baseline),
                    xmin=centroids[i] - place_field_width[i] / 2,
                    xmax=centroids[i] + place_field_width[i] / 2,
                    color='b',
                    linestyle='-',
                    linewidth=2,
                    label='peak width' if i == 0 else None
                )
        ax.legend()
        ax.set_ylim([0,ax.get_ylim()[1]])

    return place_fields


def thresholding_method_batch(behavior,neuron_activity,threshold_factor=4, sigma=4):
    """
    Batch processing for thresholding method to detect place fields.
    This function applies the PF_thresholding function to each neuron's firing rate map.
    """

    place_fields = []
    for neuron in range(neuron_activity.shape[0]):

        place_field = thresholding_method_single(behavior,neuron_activity[neuron,:],threshold_factor=threshold_factor,sigma=sigma)
        place_fields.append(place_field)
        
    return place_fields

# def PF_thresholding(firing_rate_map, threshold_factor = 4, sigma=4):

#     place_fields = {}
#     for neuron_idx, neuron_firing_map in enumerate(firing_rate_map):

#         # smooth_map = gaussian_filter1d(neuron_firing_map, sigma=sigma, mode = 'wrap')
#         # baseline = np.percentile(smooth_map, 50)
#         # _,_, std_rate = get_spikeNr(smooth_map)
#         # threshold = baseline + (threshold_factor*std_rate)

#         # #shifting the peak to center
#         # peak_index = np.argmax(smooth_map)  
#         # shift_amount = len(smooth_map) // 2 - peak_index 
#         # shifted_map = np.roll(smooth_map, shift_amount) 
    
#         # field_loc, _ = find_peaks(shifted_map, height=threshold, distance = 10, width = 2)
#         # field_loc = field_loc.tolist()

#         # PF_amplitude = []
#         # place_field_width = []
#         # centroids = []

#         if field_loc: 
#             PF_amplitude = shifted_map[field_loc].tolist()

#             widths,_,left_ips, right_ips = peak_widths(shifted_map, field_loc, rel_height= 0.6)
#             place_field_width = [int(round(width)) for width in widths]

#             #calculating centroids for each place field
#             for i,loc in enumerate(field_loc):
             
#                 left_idx = int(np.floor(left_ips[i]))
#                 right_idx = int(np.ceil(right_ips[i]))

#                 field_indices = np.arange(left_idx, right_idx + 1)
#                 field_activities = shifted_map[field_indices]

#                 #computing center of mass
#                 centroid = np.sum(field_indices * field_activities) / np.sum(field_activities)
#                 centroid = (centroid - shift_amount)% len(smooth_map)
#                 centroids.append(centroid)
            
#         place_fields[neuron_idx] = {
#         'baseline': baseline,
#         'amplitude': PF_amplitude,
#         'field_location': centroids,
#         'place_field_width': place_field_width,
#         'nbr_PF': len(centroids),
#         } 
#     return place_fields