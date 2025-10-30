import os, random
import numpy as np

from .calculate_information import *
from .information_corrections import *
# from placefield_dynamics.placefield_detection import *
# from placefield_dynamics.placefield_detection.utils import *

estimate_SI_bit_spike=1 # Choose to estimate SI (bit/spike) by setting the value to 1 (0 otherwise)
estimate_SI_bit_sec=1 # Choose to estimate SI (bit/sec) by setting the value to 1 (0 otherwise)
estimate_MI=1 # Choose to estimate MI by setting the value to 1 (0 otherwise)
measures_to_estimate=[estimate_SI_bit_spike,estimate_SI_bit_sec,estimate_MI]

settings = {
    "measures_to_estimate": measures_to_estimate,
    "dt": 1 / 15.0,
    "active_bins_threshold": 10,
    "firing_rate_threshold": 0.001,
    "estimate_only_significantly_tuned_cells": 0,
    "shuffle_type": "cyclic",
    "num_shuffles": 1000,
    "tuning_significance_threshold": 0.05,
    "subsampling_repetitions": 500,
    "subsample_fraction": np.arange(0.1, 1.1, 0.1),
    "plot_results": 1,
    "save_figures": False,
    "figures_directory": [os.path.join("figures")],
}

def estimate_unbiased_information(spike_train,stimulus_trace,settings=settings):
    '''
        This function estimates the unbiased information between the spike train and the stimulus trace.

        Inputs:
            1. spike_train - Matrix of size TxN, where T is the number of time bins and N is the number of neurons.
                Each element is the spike count of a given neuron in a given time bin.
            2. stimulus_trace - Vector of size T, where each element is the state in a
                given time bin.
            3. settings - A dictionary that contains the parameters for the analysis.
                The dictionary should contain the following keys:
                    1. measures_to_estimate - A list of strings that contains the measures to estimate.
                    The possible measures are 'MI' (mutual information), 'Isec' (specific information) and 'Ispike' (specificity).
                    2. dt - The time bin size in seconds.
                    3. active_bins_threshold - The minimum number of active bins that a neuron should have in order to be included in the analysis.
                    4. firing_rate_threshold - The minimum average firing rate that a neuron should have in order to be included in the analysis.
                
        Outputs:
            1. unbiased_information - A dictionary that contains the estimated unbiased information.
                The dictionary contains the following keys:
                    1. measures - A list of strings that contains the measures that were estimated.
                    2. MI - A vector of size N, where N is the number of neurons. Each element is the estimated mutual information of a given neuron.
                    3. Isec - A vector of size N, where N is the number of neurons. Each element is the estimated specific information of a given neuron.
                    4. Ispike - A vector of size N, where N is the number of neurons. Each element is the estimated specificity of a given neuron.
        
        The function starts here:
    '''

    measures_to_estimate = settings['measures_to_estimate']
    dt = settings['dt']
    active_bins_threshold = settings['active_bins_threshold']
    firing_rate_threshold = settings['firing_rate_threshold']

    active_bins = np.sum(spike_train>0,axis=1)
    average_firing_rates = np.mean(spike_train,axis=1)/dt

    sufficiently_active_cells_indexes = np.where(np.logical_and(active_bins>=active_bins_threshold,average_firing_rates>firing_rate_threshold))[0]
    fraction_sufficiently_active_cells = len(sufficiently_active_cells_indexes) / len(
        active_bins
    )

    if len(sufficiently_active_cells_indexes) > 0:

        if measures_to_estimate[0] or measures_to_estimate[1]:
            # Computing the tuning curves of the cells:
            tuning_curves, normalized_states_distribution = compute_tuning_curves(spike_train, stimulus_trace, dt)
            # naive SI:
            SI_naive_bit_spike, SI_naive_bit_sec = compute_SI(average_firing_rates[sufficiently_active_cells_indexes], tuning_curves[sufficiently_active_cells_indexes, :], normalized_states_distribution)

        if measures_to_estimate[2]:
            MI_naive = compute_MI(spike_train[sufficiently_active_cells_indexes,:], stimulus_trace)

        if settings['estimate_only_significantly_tuned_cells']:
            shuffle_type = settings['shuffle_type']
            num_shuffles = settings['num_shuffles']
            tuning_significance_threshold = settings['tuning_significance_threshold']

            # obtaining shuffled spike trains:
            shuffled_spike_trains = shuffle_spike_trains(spike_train[sufficiently_active_cells_indexes,:], num_shuffles, shuffle_type)

            # Identifying significantly modulated cells
            if measures_to_estimate[0] or measures_to_estimate[1]:  # based on the SI in active cells for naive versus shuffle

                # Shuffle SI
                SI_shuffle_bit_spike = np.empty((len(sufficiently_active_cells_indexes), num_shuffles))
                for i in range(num_shuffles):
                    temp_shuffled_tuning_curves, _ = compute_tuning_curves(shuffled_spike_trains[:, :, i], stimulus_trace, dt)
                    SI_shuffle_bit_spike[:, i], _ = compute_SI(average_firing_rates[sufficiently_active_cells_indexes], temp_shuffled_tuning_curves, normalized_states_distribution)

                tuning_significance_active_cells = 1 - np.sum(np.tile(SI_naive_bit_spike[:,np.newaxis], (1, num_shuffles)) > SI_shuffle_bit_spike, axis=1) / num_shuffles

                p_value_significantly_tuned_and_active_cells = tuning_significance_active_cells[tuning_significance_active_cells < tuning_significance_threshold]

                significantly_tuned_and_active_cells_indexes = sufficiently_active_cells_indexes[tuning_significance_active_cells < tuning_significance_threshold]

                SI_naive_bit_spike_significantly_tuned_cells = SI_naive_bit_spike[tuning_significance_active_cells < tuning_significance_threshold]

                SI_naive_bit_sec_significantly_tuned_cells = SI_naive_bit_sec[tuning_significance_active_cells < tuning_significance_threshold]

                if len(significantly_tuned_and_active_cells_indexes) > 0:
                    average_SI_naive_bit_spike_significantly_tuned_cells = np.nanmean(SI_naive_bit_spike_significantly_tuned_cells, axis=0)
                    average_SI_naive_bit_sec_significantly_tuned_cells = np.nanmean(SI_naive_bit_sec_significantly_tuned_cells, axis=0)

            elif measures_to_estimate[2]:
                # Shuffle MI
                MI_shuffle = np.empty((len(sufficiently_active_cells_indexes), num_shuffles))
                for n in range(num_shuffles):
                    MI_shuffle[:, n] = compute_MI(shuffled_spike_trains[:, :, n], stimulus_trace)

                tuning_significance_active_cells = 1 - np.sum(np.tile(MI_naive, (1, num_shuffles)) > MI_shuffle, axis=1) / num_shuffles

                p_value_significantly_tuned_and_active_cells = tuning_significance_active_cells[tuning_significance_active_cells < tuning_significance_threshold]

                MI_naive_significantly_tuned_cells = MI_naive[tuning_significance_active_cells < tuning_significance_threshold]

                if len(MI_naive_significantly_tuned_cells) > 0:
                    average_MI_naive_significantly_tuned_cells = np.mean(MI_naive_significantly_tuned_cells, axis=0, nan=True)
                    significantly_tuned_and_active_cells_indexes = sufficiently_active_cells_indexes[tuning_significance_active_cells < tuning_significance_threshold]

            if (measures_to_estimate[0] or measures_to_estimate[1]) and measures_to_estimate[2]:
                MI_naive_significantly_tuned_cells = MI_naive[tuning_significance_active_cells < tuning_significance_threshold]
                if len(MI_naive_significantly_tuned_cells) > 0:
                    average_MI_naive_significantly_tuned_cells = np.nanmean(MI_naive_significantly_tuned_cells, axis=0)

        else:
            significantly_tuned_and_active_cells_indexes = sufficiently_active_cells_indexes
            if measures_to_estimate[0] or measures_to_estimate[1]:
                SI_naive_bit_spike_significantly_tuned_cells = SI_naive_bit_spike
                SI_naive_bit_sec_significantly_tuned_cells = SI_naive_bit_sec
                if len(SI_naive_bit_spike_significantly_tuned_cells) > 0:
                    average_SI_naive_bit_spike_significantly_tuned_cells = np.nanmean(SI_naive_bit_spike_significantly_tuned_cells)
                    average_SI_naive_bit_sec_significantly_tuned_cells = np.nanmean(SI_naive_bit_sec_significantly_tuned_cells)
            elif measures_to_estimate[2]:
                MI_naive_significantly_tuned_cells = MI_naive
                if len(MI_naive_significantly_tuned_cells) > 0:
                    average_MI_naive_significantly_tuned_cells = np.nanmean(MI_naive_significantly_tuned_cells)

            if (measures_to_estimate[0] or measures_to_estimate[1]) and measures_to_estimate[2]:
                MI_naive_significantly_tuned_cells = MI_naive
                if len(MI_naive_significantly_tuned_cells) > 0:
                    average_MI_naive_significantly_tuned_cells = np.nanmean(MI_naive_significantly_tuned_cells)

        if len(significantly_tuned_and_active_cells_indexes)>0:
            average_rates_significantly_tuned_cells = average_firing_rates[significantly_tuned_and_active_cells_indexes]
            active_bins_significantly_tuned_cells = active_bins[significantly_tuned_and_active_cells_indexes]
            fraction_significantly_tuned_and_active_cells = len(significantly_tuned_and_active_cells_indexes) / len(active_bins)
            fraction_of_significantly_tuned_from_active_cells = len(significantly_tuned_and_active_cells_indexes) / len(sufficiently_active_cells_indexes)
            spike_train_significantly_tuned_cells = spike_train[significantly_tuned_and_active_cells_indexes, :]

            if settings['estimate_only_significantly_tuned_cells']:
                print(f"Found {len(sufficiently_active_cells_indexes)}/{len(active_bins)} sufficiently active cells, out of which {len(significantly_tuned_and_active_cells_indexes)} cells ({round(100*len(significantly_tuned_and_active_cells_indexes)/len(sufficiently_active_cells_indexes))}% are significantly modulated by the encoded variable.")
            else:
                print(f"Found {len(sufficiently_active_cells_indexes)}/{len(active_bins)} sufficiently active cells")

            subsampling_repetitions = settings['subsampling_repetitions']
            T = spike_train.shape[1]
            subsample_size = settings['subsample_fraction'] * T
            print('subsample_size:',subsample_size)

            print('Computing information as a function of subsample size:')
            if measures_to_estimate[0] or measures_to_estimate[1]:
                if measures_to_estimate[2]:
                    SI_naive_bit_spike_versus_sample_size, SI_shuffle_bit_spike_versus_sample_size, SI_naive_bit_sec_versus_sample_size, SI_shuffle_bit_sec_versus_sample_size, MI_naive_versus_sample_size, MI_shuffle_versus_sample_size = compute_information_versus_sample_size(spike_train_significantly_tuned_cells, stimulus_trace, subsample_size, dt, subsampling_repetitions, measures_to_estimate)
                else:
                    SI_naive_bit_spike_versus_sample_size, SI_shuffle_bit_spike_versus_sample_size, SI_naive_bit_sec_versus_sample_size, SI_shuffle_bit_sec_versus_sample_size = compute_information_versus_sample_size(spike_train_significantly_tuned_cells, stimulus_trace, subsample_size, dt, subsampling_repetitions, measures_to_estimate)
            elif measures_to_estimate[2]:
                _, _, _, _, MI_naive_versus_sample_size, MI_shuffle_versus_sample_size = compute_information_versus_sample_size(spike_train_significantly_tuned_cells, stimulus_trace, subsample_size, dt, subsampling_repetitions, measures_to_estimate)

            # Correcting the bias using the SSR BAE methods:
            plot_results = settings['plot_results']
            save_figures = settings['save_figures']
            figures_directory = settings['figures_directory']
            if measures_to_estimate[0]:
                units = 'bit/spike'

                # SSR method:
                SI_SSR_bit_spike, average_SI_SSR_bit_spike, SI_SSR_stability_bit_spike, average_SI_SSR_stability_bit_spike = perform_SSR(SI_naive_bit_spike_versus_sample_size, SI_shuffle_bit_spike_versus_sample_size, subsample_size, units, plot_results, save_figures, figures_directory)

                # BAE method:
                SI_BAE_bit_spike, average_SI_BAE_bit_spike, SI_BAE_fit_R_2_bit_spike, average_SI_BAE_fit_R_2_bit_spike = perform_BAE(SI_naive_bit_spike_versus_sample_size, subsample_size, units, plot_results, save_figures, figures_directory)

                SI_disagreement_bit_spike = SI_BAE_bit_spike - SI_SSR_bit_spike
                average_SI_disagreement_bit_spike = average_SI_BAE_bit_spike - average_SI_SSR_bit_spike

            # Correcting the bias for SI in bit/sec:
            if measures_to_estimate[1]:
                units = 'bit/sec'

                # SSR method:
                SI_SSR_bit_sec, average_SI_SSR_bit_sec, SI_SSR_stability_bit_sec, average_SI_SSR_stability_bit_sec = perform_SSR(SI_naive_bit_sec_versus_sample_size, SI_shuffle_bit_sec_versus_sample_size, subsample_size, units, plot_results, save_figures, figures_directory)

                # BAE method:
                SI_BAE_bit_sec, average_SI_BAE_bit_sec, SI_BAE_fit_R_2_bit_sec, average_SI_BAE_fit_R_2_bit_sec = perform_BAE(SI_naive_bit_sec_versus_sample_size, subsample_size, units, plot_results, save_figures, figures_directory)

                SI_disagreement_bit_sec = SI_BAE_bit_sec - SI_SSR_bit_sec
                average_SI_disagreement_bit_sec = average_SI_BAE_bit_sec - average_SI_SSR_bit_sec

            # Correcting the bias for MI:
            if measures_to_estimate[2]:
                units = 'bit'

                # SSR method:
                MI_SSR, average_MI_SSR, MI_SSR_stability, average_MI_SSR_stability = perform_SSR(MI_naive_versus_sample_size, MI_shuffle_versus_sample_size, subsample_size, units, plot_results, save_figures, figures_directory)

                # BAE method:
                MI_BAE, average_MI_BAE, MI_BAE_fit_R_2, average_MI_BAE_fit_R_2 = perform_BAE(MI_naive_versus_sample_size, subsample_size, units, plot_results, save_figures, figures_directory)

                MI_disagreement = MI_BAE - MI_SSR
                average_MI_disagreement = average_MI_BAE - average_MI_SSR

            if plot_results or save_figures:
                if measures_to_estimate[0]:  # for SI in bit/spike
                    # Cross validation:
                    if plot_results:
                        plt.figure()
                    else:
                        plt.figure(visible=False)
                    plt.plot(SI_SSR_bit_spike, SI_BAE_bit_spike, '.', markersize=15, color='b')
                    plt.plot([0, 1.1 * np.max(SI_BAE_bit_spike)], [0, 1.1 * np.max(SI_BAE_bit_spike)], '--k', linewidth=2)
                    plt.xlim([0, 1.1 * np.max(SI_BAE_bit_spike)])
                    plt.ylim([0, 1.1 * np.max(SI_BAE_bit_spike)])
                    plt.axis('square')
                    plt.xlabel('SSR estimation (bit/spike)')
                    plt.ylabel('BAE estimation (bit/spike)')
                    plt.title('SI (bit/spike)')
                    # plt.xticks(fontsize=16)
                    # plt.yticks(fontsize=16)
                    plt.box(False)
                    if save_figures:
                        plt.savefig(os.path.join(figures_directory, 'BAE versus SSR - SI bit per spike.fig'))
                        plt.savefig(os.path.join(figures_directory, 'BAE versus SSR - SI bit per spike.png'))

                    # Estimation quality
                    if plot_results:
                        plt.figure()
                    else:
                        plt.figure(visible=False)
                    plt.plot(SI_SSR_stability_bit_spike, SI_BAE_fit_R_2_bit_spike, '.', markersize=15, color='b')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.axis('square')
                    plt.xlabel('SSR stability')
                    plt.ylabel('BAE fit R^2')
                    plt.title('SI (bit/spike)')
                    # plt.xticks(fontsize=16)
                    # plt.yticks(fontsize=16)
                    plt.box(False)
                    if save_figures:
                        plt.savefig(os.path.join(figures_directory, 'Estimation quality - SI bit per spike.fig'))
                        plt.savefig(
                            os.path.join(
                                figures_directory,
                                "Estimation quality - SI bit per spike.png",
                            )
                        )

                if measures_to_estimate[1]: # for SI in bit/sec

                    # Cross validation:
                    if plot_results:
                        plt.figure()
                    else:
                        plt.figure(visible=False)

                    plt.plot(SI_SSR_bit_sec, SI_BAE_bit_sec, '.', markersize=15, color='b')
                    plt.plot([0, 1.1 * np.max(SI_BAE_bit_sec)], [0, 1.1 * np.max(SI_BAE_bit_sec)], '--k', linewidth=2)
                    plt.xlim([0, 1.1 * np.max(SI_BAE_bit_sec)])
                    plt.ylim([0, 1.1 * np.max(SI_BAE_bit_sec)])
                    plt.axis('square')
                    plt.xlabel('SSR estimation (bit/sec)')
                    plt.ylabel('BAE estimation (bit/sec)')
                    plt.title('SI (bit/sec)')
                    plt.tick_params(labelsize=16)
                    plt.box(False)
                    if save_figures:
                        plt.savefig(os.path.join(figures_directory, 'BAE versus SSR - SI bit per sec.fig'))
                        plt.savefig(os.path.join(figures_directory, 'BAE versus SSR - SI bit per sec.png'))

                    # Estimation quality
                    if plot_results:
                        plt.figure()
                    else:
                        plt.figure(visible=False)

                    plt.plot(SI_SSR_stability_bit_sec, SI_BAE_fit_R_2_bit_sec, '.', markersize=15, color='b')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.axis('square')
                    plt.xlabel('SSR stability')
                    plt.ylabel('BAE fit R^2')
                    plt.title('SI (bit/sec)')
                    plt.tick_params(labelsize=16)
                    plt.box(False)
                    if save_figures:
                        plt.savefig(os.path.join(figures_directory, 'Estimation quality - SI bit per sec.fig'))
                        plt.savefig(
                            os.path.join(
                                figures_directory,
                                "Estimation quality - SI bit per sec.png",
                            )
                        )

                if measures_to_estimate[2]: # for MI
                    # Cross validation:
                    if plot_results:
                        plt.figure()
                    else:
                        plt.figure(visible=False)
                    plt.plot(MI_SSR, MI_BAE, '.', markersize=15, color='b')
                    plt.plot([0, 1.1 * np.max(MI_BAE)], [0, 1.1 * np.max(MI_BAE)], '--k', linewidth=2)
                    plt.xlim([0, 1.1 * np.max(MI_BAE)])
                    plt.ylim([0, 1.1 * np.max(MI_BAE)])
                    plt.axis('square')
                    plt.xlabel('SSR estimation (bit)')
                    plt.ylabel('BAE estimation (bit)')
                    plt.title('MI')
                    # plt.tick_params(labelsize=16)
                    plt.box(False)
                    if save_figures:
                        plt.savefig(os.path.join(figures_directory, 'BAE versus SSR - MI.fig'))
                        plt.savefig(os.path.join(figures_directory, 'BAE versus SSR - MI.png'))

                    # Estimation quality
                    if plot_results:
                        plt.figure()
                    else:
                        plt.figure(visible=False)
                    plt.plot(MI_SSR_stability, MI_BAE_fit_R_2, '.', markersize=15, color='b')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.axis('square')
                    plt.xlabel('SSR stability')
                    plt.ylabel('BAE fit R^2')
                    plt.title('MI')
                    # plt.tick_params(labelsize=16)
                    plt.box(False)
                    if save_figures:
                        plt.savefig(os.path.join(figures_directory, 'Estimation quality - MI.fig'))
                        plt.savefig(
                            os.path.join(
                                figures_directory, "Estimation quality - MI.png"
                            )
                        )

            if not settings['estimate_only_significantly_tuned_cells']:
                shuffle_type = settings['shuffle_type']
                num_shuffles = settings['num_shuffles']
                tuning_significance_threshold = settings['tuning_significance_threshold']

                # obtaining shuffled spike trains:
                shuffled_spike_trains = shuffle_spike_trains(spike_train[sufficiently_active_cells_indexes,:], num_shuffles, shuffle_type)

                # Identifying significantly modulated cells:
                if measures_to_estimate[0] or measures_to_estimate[1]: # based on the SI in active cells for naive versus shuffle

                    # shuffle SI:
                    SI_shuffle_bit_spike = np.empty((len(sufficiently_active_cells_indexes), num_shuffles))
                    # display_progress_bar('Computing shuffle information for the tuning significance test: ', False)
                    for n in range(num_shuffles):
                        # display_progress_bar(100*(n/num_shuffles), False)
                        temp_shuffled_tuning_curves, _ = compute_tuning_curves(shuffled_spike_trains[:,:,n], stimulus_trace, dt)
                        SI_shuffle_bit_spike[:,n], _ = compute_SI(average_firing_rates[sufficiently_active_cells_indexes], temp_shuffled_tuning_curves, normalized_states_distribution)
                    # display_progress_bar(' done', False)
                    # display_progress_bar('', True)

                    # Finding significantly tuned cells:
                    tuning_significance_active_cells = 1 - np.sum(np.tile(SI_naive_bit_spike[:,np.newaxis], (1, num_shuffles)) > SI_shuffle_bit_spike, axis=1) / num_shuffles
                    p_value_significantly_tuned_and_active_cells = tuning_significance_active_cells[tuning_significance_active_cells < tuning_significance_threshold]
                    significantly_tuned_and_active_cells_indexes = sufficiently_active_cells_indexes[tuning_significance_active_cells < tuning_significance_threshold]
                elif measures_to_estimate[2]: # based on the MI in active cells for naive versus shuffle

                    # shuffle MI:
                    MI_shuffle = np.empty((len(sufficiently_active_cells_indexes), num_shuffles))
                    # display_progress_bar('Computing shuffle information: ', False)
                    for n in range(num_shuffles):
                        # display_progress_bar(100*(n/num_shuffles), False)
                        MI_shuffle[:,n] = compute_MI(np.squeeze(shuffled_spike_trains[:,:,n]), stimulus_trace)
                    # display_progress_bar(' done', False)
                    # display_progress_bar('', True)

                    # Finding significantly tuned cells:
                    tuning_significance_active_cells = 1 - np.sum(np.tile(MI_naive, (1, num_shuffles)) > MI_shuffle, axis=1) / num_shuffles
                    p_value_significantly_tuned_and_active_cells = tuning_significance_active_cells[tuning_significance_active_cells < tuning_significance_threshold]
                    if len(MI_naive_significantly_tuned_cells) > 0:
                        significantly_tuned_and_active_cells_indexes = sufficiently_active_cells_indexes[tuning_significance_active_cells < tuning_significance_threshold]

            unbiased_information_estimation_results = {}

            # General parameters
            unbiased_information_estimation_results['settings'] = {
                'dt': dt,
                'num_shuffles': num_shuffles,
                'shuffle_type': shuffle_type,
                'tuning_significance_threshold': tuning_significance_threshold,
                'active_bins_threshold': active_bins_threshold,
                'firing_rate_threshold': firing_rate_threshold,
                'subsampling_repetitions': subsampling_repetitions
            }

            # Data statistics
            unbiased_information_estimation_results['firing_statistics'] = {
                'fraction_sufficiently_active_cells': fraction_sufficiently_active_cells,
                'fraction_significantly_tuned_and_active_cells': fraction_significantly_tuned_and_active_cells,
                'fraction_of_significantly_tuned_from_active_cells': fraction_of_significantly_tuned_from_active_cells,
                'average_rates': average_rates_significantly_tuned_cells,
                'active_bins': active_bins_significantly_tuned_cells,
                'sufficiently_active_cells_indexes': sufficiently_active_cells_indexes,
                'significantly_tuned_and_active_cells_indexes': significantly_tuned_and_active_cells_indexes,
                'p_value_significantly_tuned_and_active_cells': p_value_significantly_tuned_and_active_cells
            }

            # Estimated information
            if measures_to_estimate[0]:  # SI in bit/spike
                # For individual cells
                unbiased_information_estimation_results['information'] = {
                    'SI_naive_bit_spike': SI_naive_bit_spike_significantly_tuned_cells,
                    'SI_SSR_bit_spike': SI_SSR_bit_spike,
                    'SI_BAE_bit_spike': SI_BAE_bit_spike,
                    'SI_disagreement_bit_spike': SI_disagreement_bit_spike,
                    'SI_SSR_stability_bit_spike': SI_SSR_stability_bit_spike,
                    'SI_BAE_fit_R_2_bit_spike': SI_BAE_fit_R_2_bit_spike
                }

                # Average across the population
                unbiased_information_estimation_results['information']['average_SI_naive_bit_spike'] = average_SI_naive_bit_spike_significantly_tuned_cells
                unbiased_information_estimation_results['information']['average_SI_SSR_bit_spike'] = average_SI_SSR_bit_spike
                unbiased_information_estimation_results['information']['average_SI_BAE_bit_spike'] = average_SI_BAE_bit_spike
                unbiased_information_estimation_results['information']['average_SI_disagreement_bit_spike'] = average_SI_disagreement_bit_spike
                unbiased_information_estimation_results['information']['average_SI_SSR_stability_bit_spike'] = average_SI_SSR_stability_bit_spike
                unbiased_information_estimation_results['information']['average_SI_BAE_fit_R_2_bit_spike'] = average_SI_BAE_fit_R_2_bit_spike

                # Average across spikes (weighted by the cells firing rates)
                unbiased_information_estimation_results['information']['weighted_average_SI_naive_bit_spike'] = sum(SI_naive_bit_spike_significantly_tuned_cells * average_rates_significantly_tuned_cells) / sum(average_rates_significantly_tuned_cells)
                unbiased_information_estimation_results['information']['weighted_average_SI_SSR_bit_spike'] = sum(SI_SSR_bit_spike * average_rates_significantly_tuned_cells) / sum(average_rates_significantly_tuned_cells)
                unbiased_information_estimation_results['information']['weighted_average_SI_BAE_bit_spike'] = sum(SI_BAE_bit_spike * average_rates_significantly_tuned_cells) / sum(average_rates_significantly_tuned_cells)

            if measures_to_estimate[1]:  # SI in bit/sec
                # For individual cells
                unbiased_information_estimation_results['information']['SI_naive_bit_sec'] = SI_naive_bit_sec_significantly_tuned_cells
                unbiased_information_estimation_results['information']['SI_SSR_bit_sec'] = SI_SSR_bit_sec
                unbiased_information_estimation_results['information']['SI_BAE_bit_sec'] = SI_BAE_bit_sec
                unbiased_information_estimation_results['information']['SI_disagreement_bit_sec'] = SI_disagreement_bit_sec
                unbiased_information_estimation_results['information']['SI_SSR_stability_bit_sec'] = SI_SSR_stability_bit_sec
                unbiased_information_estimation_results['information']['SI_BAE_fit_R_2_bit_sec'] = SI_BAE_fit_R_2_bit_sec

                # Average across the population
                unbiased_information_estimation_results['information']['average_SI_naive_bit_sec'] = average_SI_naive_bit_sec_significantly_tuned_cells
                unbiased_information_estimation_results['information']['average_SI_SSR_bit_sec'] = average_SI_SSR_bit_sec
                unbiased_information_estimation_results['information']['average_SI_BAE_bit_sec'] = average_SI_BAE_bit_sec
                unbiased_information_estimation_results['information']['average_SI_disagreement_bit_sec'] = average_SI_disagreement_bit_sec
                unbiased_information_estimation_results['information']['average_SI_SSR_stability_bit_sec'] = average_SI_SSR_stability_bit_sec
                unbiased_information_estimation_results['information']['average_SI_BAE_fit_R_2_bit_sec'] = average_SI_BAE_fit_R_2_bit_sec

            if measures_to_estimate[2]: # MI
                # For individual cells:
                unbiased_information_estimation_results['information']['MI_naive'] = MI_naive_significantly_tuned_cells
                unbiased_information_estimation_results['information']['MI_SSR'] = MI_SSR
                unbiased_information_estimation_results['information']['MI_BAE'] = MI_BAE
                unbiased_information_estimation_results['information']['MI_disagreement'] = MI_disagreement
                unbiased_information_estimation_results['information']['MI_SSR_stability'] = MI_SSR_stability
                unbiased_information_estimation_results['information']['MI_BAE_fit_R_2'] = MI_BAE_fit_R_2

                # Average across the population:
                unbiased_information_estimation_results['information']['average_MI_naive'] = average_MI_naive_significantly_tuned_cells
                unbiased_information_estimation_results['information']['average_MI_SSR'] = average_MI_SSR
                unbiased_information_estimation_results['information']['average_MI_BAE'] = average_MI_BAE
                unbiased_information_estimation_results['information']['average_MI_disagreement'] = average_MI_disagreement
                unbiased_information_estimation_results['information']['average_MI_SSR_stability'] = average_MI_SSR_stability
                unbiased_information_estimation_results['information']['average_MI_BAE_fit_R_2'] = average_MI_BAE_fit_R_2

            print('Finished analyzing data set')
        else:
            print('Found', len(sufficiently_active_cells_indexes), '/', len(active_bins), 'sufficiently active cells')
            print('No significantly tuned cells were found')
    else:
        print('No sufficiently active cells were found')

    return unbiased_information_estimation_results


def compute_tuning_curves(spike_train, stimulus_trace, dt):
    """
        Computes the tuning curves and stimulus distribution.

        Inputs:
        1. spike_train - Matrix of size TxN, where T is the number of time bins
            and N is the number of neurons. Each element is the spike count of a
            given neuron in a given time bin.
        2. stimulus_trace - Vector of size T, where each element is the state in a
            given time bin.
        3. dt - Temporal bin size (in seconds)

        Outputs:
        1. tuning_curves - Matrix of size NxS with the firing rate map of each
            neuron for each stimulus value.
        2. stimulus_distribution - Vector of size S with the probabilities of the
            different stimuli.

    """

    #calculating stimulus distribution
    stimulus_values, stimulus_counts = np.unique(stimulus_trace, return_counts=True)
    stimulus_distribution = stimulus_counts / len(stimulus_trace)

    #determining the number of stimulus values, neurons, and time bins
    num_stimulus_values = len(stimulus_values)
    num_cells = spike_train.shape[0]

    #intilaizing tuning_curves
    tuning_curves = np.zeros((num_cells, num_stimulus_values))#, order = 'F')

    for i in range(num_stimulus_values):
        this_bin_indexes = np.where(stimulus_trace == stimulus_values[i])[0]

        if len(this_bin_indexes) > 1:
            #calculating the mean firing rate within the time bin
            tuning_curves[:, i] = np.mean(spike_train[:, this_bin_indexes], axis=1) / dt
        else:
            #using the firing rate directly if only one time bin
            tuning_curves[:, i] = spike_train[:, this_bin_indexes].flatten() / dt

    return tuning_curves, stimulus_distribution


def shuffle_spike_trains(spike_train, num_shuffles, shuffle_type,**kwargs):
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
    elif shuffle_type == 'cyclic_trials':
        
        assert 'trials' in kwargs, 'Please provide the trials dictionary'
        trials = kwargs['trials']

        for n in range(num_shuffles):
        
            shuffled_spike_trains[:, :, n] = np.roll(
                np.hstack(
                    [
                        np.roll(spike_train[:,trials['start'][t]:trials['start'][t+1]],int(random.random()*trials['nFrames'][t]),axis=1)
                            for t in np.random.permutation(trials['ct'])
                    ]
                ),
                int(random.random()*spike_train.shape[1]),
                axis=1
            )
    elif shuffle_type == 'random':
        for n in range(num_shuffles):
            random_indexes = np.random.permutation(T)  #generating random indexes
            shuffled_spike_trains[:, :, n] = spike_train[:, random_indexes]  #random shuffling the spike train
    else:
        raise ValueError('Please choose a valid shuffling type')
    

    return shuffled_spike_trains


def compute_information_versus_sample_size(spike_train, stimulus_trace, sample_sizes, dt, repetitions, info_measures):

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
    if spike_train.dtype != int:
        spikes = np.copy(spike_train)
        spikes[spikes == 0] = np.nan
        thr = np.nanmedian(spikes, axis=1)
        thr[np.isnan(thr)] = np.nanmean(thr)
        spike_train = np.clip(np.ceil(spike_train / thr[:, None]),a_min=0,a_max=30).astype('int')

    # sample_sizes = sample_sizes*T
    nbr_samples = len(sample_sizes)

    # initializing arrays to store information content
    if info_measures[0] or info_measures[1]:
        info_bit_spike_vs_sample = np.full((N, nbr_samples), np.nan, order = 'F')
        shuffle_info_bit_spike_vs_sample = np.full((N, nbr_samples), np.nan, order = 'F')
        info_bit_sec_vs_sample = np.full((N, nbr_samples), np.nan, order = 'F')
        shuffle_info_bit_sec_vs_sample = np.full((N, nbr_samples), np.nan, order = 'F')

    if info_measures[2]:
        info_mi_vs_sample = np.full((N,nbr_samples), np.nan, order = 'F')
        shuffle_info_mi_vs_sample = np.full((N, nbr_samples), np.nan, order = 'F')

    t_shuffle = 0
    t_SI = 0
    t_MI = 0
    # calculating info for different sample sizes
    for n in range(nbr_samples):

        col_dim = int(np.ceil(repetitions * T / sample_sizes[n]))
        print(col_dim)

        num_time_bins = int(np.floor(sample_sizes[n]))

        if info_measures[0] or info_measures[1]:
            # initializing arrays to store information content
            info_bit_spike = np.full((N, col_dim), np.nan, order = 'F')
            shuffle_info_bit_spike = np.full((N, col_dim), np.nan, order = 'F')
            info_bit_sec = np.full((N, col_dim), np.nan, order = 'F')
            shuffle_info_bit_sec = np.full((N, col_dim), np.nan, order = 'F')

        if info_measures[2]:
            # initializing arrays to store information content
            info_mi = np.full((N, col_dim), np.nan, order = 'F')
            shuffle_info_mi = np.full((N, col_dim), np.nan, order = 'F')

        for k in range(col_dim):
            # shuffling spike trains
            # sample_indexes = np.argsort(np.random.rand(T))[:num_time_bins]

            t_shuffle_start = time.time()
            sample_indexes = np.random.permutation(T)[:num_time_bins]
            shuffled_spikes =np.squeeze( shuffle_spike_trains(spike_train, 1, 'cyclic'))
            shuffled_spikes =np.squeeze( shuffle_spike_trains(spike_train[:,sample_indexes], 1, 'cyclic'))

            spike_train_sample = spike_train[:, sample_indexes]
            stimulus_trace_sample = stimulus_trace[sample_indexes]

            t_shuffle += time.time() - t_shuffle_start

            if info_measures[0] or info_measures[1]:

                t_SI_start = time.time()
                # computing tunung curves and calculating information content
                temp_tc, temp_states_distribution = compute_tuning_curves(spike_train_sample, stimulus_trace_sample, dt)
                temp_fr = np.mean(spike_train_sample, axis=1) / dt

                temp_info_bit_spike, temp_info_bit_sec = compute_SI(temp_fr, temp_tc, temp_states_distribution)

                info_bit_spike[:, k] = temp_info_bit_spike
                info_bit_sec[:, k] = temp_info_bit_sec

                temp_shuffled_tc, _ = compute_tuning_curves(shuffled_spikes, stimulus_trace_sample, dt)
                temp_shuffle_fr = np.mean(shuffled_spikes, axis=1) / dt
                temp_shuffle_info_bit_spike, temp_shuffle_info_bit_sec = compute_SI(temp_shuffle_fr, temp_shuffled_tc, temp_states_distribution)
                shuffle_info_bit_spike[:, k] = temp_shuffle_info_bit_spike
                shuffle_info_bit_sec[:, k] = temp_shuffle_info_bit_sec

                t_SI += time.time() - t_SI_start

            if info_measures[2]:

                t_MI_start = time.time()
                # print(spike_train_sample,spike_train_sample.max())
                # print(spike_train_sample.shape, stimulus_trace_sample.shape)
                temp_mi = compute_MI(spike_train_sample, stimulus_trace_sample)
                # return
                info_mi[:, k] = temp_mi

                temp_mi_shuffle = compute_MI(shuffled_spikes, stimulus_trace_sample)
                shuffle_info_mi[:, k] = temp_mi_shuffle
                t_MI += time.time() - t_MI_start

        if info_measures[0] or info_measures[1]:
            # averaging info content across sample sizes
            info_bit_spike_vs_sample[:, n] = np.nanmean(info_bit_spike, axis=1)
            shuffle_info_bit_spike_vs_sample[:, n] = np.nanmean(shuffle_info_bit_spike, axis=1)
            info_bit_sec_vs_sample[:, n] = np.nanmean(info_bit_sec, axis=1)
            shuffle_info_bit_sec_vs_sample[:, n] = np.nanmean(shuffle_info_bit_sec, axis=1)

        if info_measures[2]:
            info_mi_vs_sample[:, n] = np.nanmean(info_mi, axis=1)
            shuffle_info_mi_vs_sample[:, n] = np.nanmean(shuffle_info_mi, axis=1)

    print('times: (shuffle/SI/MI)',t_shuffle, t_SI, t_MI)

    results = []
    if info_measures[0] or info_measures[1]:
        results.extend([info_bit_spike_vs_sample, shuffle_info_bit_spike_vs_sample, info_bit_sec_vs_sample, shuffle_info_bit_sec_vs_sample])
    if info_measures[2]:
        results.extend([info_mi_vs_sample, shuffle_info_mi_vs_sample])

    return results      
