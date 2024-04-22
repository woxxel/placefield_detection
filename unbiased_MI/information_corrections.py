import os
import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def perform_BAE(information_versus_sample_size, subsample_size, units, plot_results, save_figures, figures_directory):
    
    '''
        This functions corrects the upward bias in the naive calculation of
        information content for limited sample sizes using the bounded asymptotic extrapolation (BAE) method.
        BAE is based on fitting the function of how the information changes with sample size and extrapolating it to infinity.
        The obtained results are compared against the unbounded aymptotic extrapolation (AE) method.

        Inputs:
            1. information_versus_sample_size - Matrix of size TxN with the estimated
            information of each of N neurons as a function of T different sample
            sizes.
            2. subsample_size - Vector of T different sample sizes
            3. units - Either bit/spike, bit/sec, or bit
            4. plot_results - 1 for plotting and 0 for not
            5. save_figures - 1 for saving the figures and 0 for not
            6. figures_directory - Path for saving the figures


        Outputs:
            1. BAE_information - Vector of size N with the estimated
            information for each neuron.
            2. average_BAE_information - Single value with the estimated
            average information for the population.
            3. BAE_fit_R_2 - Vector of size N with the squared residuals of the
            fitted curve for each neuron
            4. average_BAE_fit_R_2 - Single value with the average squared residuals for the population.
    '''

    # extrapolating the information for the average average with information(t)=a+b/(1+ct) - BAE:
    middle_index = round(len(subsample_size) / 2)
    middle_sample_size = subsample_size[middle_index]
    
    if information_versus_sample_size.shape[0] > 1:
        average_information_versus_sample_size = np.nanmean(information_versus_sample_size, axis=0)
        a_0 = np.nanmean(information_versus_sample_size[:, -1])
        b_0 = np.nanmean(information_versus_sample_size[:, 0]) - np.nanmean(information_versus_sample_size[:, -1])
        c_0 = 1 / middle_sample_size * (np.nanmean(information_versus_sample_size[:, 0]) - np.nanmean(information_versus_sample_size[:, middle_index])) / (np.nanmean(information_versus_sample_size[:, middle_index]) - np.nanmean(information_versus_sample_size[:, -1]))
    else:
        average_information_versus_sample_size = information_versus_sample_size
        a_0 = information_versus_sample_size[0, -1]
        b_0 = information_versus_sample_size[0, 0] - information_versus_sample_size[0, -1]
        c_0 = 1 / middle_sample_size * (information_versus_sample_size[0, 0] - information_versus_sample_size[0, middle_index]) / (information_versus_sample_size[0, middle_index] - information_versus_sample_size[0, -1])
    
    initial_parameters = [a_0, b_0, c_0]
    
    def F_model(x, p1,p2,p3):
        return p1 + (p2) / (1 + p3 * x)
    
    lb = [0, 0, 0]
    ub = [np.inf, np.inf, np.inf]
    
    options = {'maxiter': 1000, 'maxfev': 2000}

    # finding the parameters that best fit the data (BAE):
    average_BAE_fit_params, _ = curve_fit(F_model, subsample_size, average_information_versus_sample_size, bounds=(lb, ub), method='trf')
    BAE_fitted_model = average_BAE_fit_params[0] + average_BAE_fit_params[1] / (1 + average_BAE_fit_params[2] * subsample_size)
    average_BAE_information = average_BAE_fit_params[0]
    average_BAE_fit_R_2 = 1 - np.mean((BAE_fitted_model - average_information_versus_sample_size) ** 2) / np.var(average_information_versus_sample_size)

    # extrapolating the information for the average average with information(t)=a+b/t+c/t^2 - AE:
    if information_versus_sample_size.shape[0] > 1:
        a_0 = np.mean(information_versus_sample_size[:, -1], axis=0)
        b_0 = np.mean(information_versus_sample_size[:, 0], axis=0) - np.mean(information_versus_sample_size[:, -1], axis=0)
        c_0 = np.mean(information_versus_sample_size[:, 0], axis=0) - np.mean(information_versus_sample_size[:, -1], axis=0)
    else:
        a_0 = information_versus_sample_size[0, -1]
        b_0 = information_versus_sample_size[0, 0] - information_versus_sample_size[0, -1]
        c_0 = information_versus_sample_size[0, 0] - information_versus_sample_size[0, -1]
    initial_parameters = [a_0, b_0, c_0]

    # def F_model(x, xdata):
    def F_model(x, p1,p2,p3):
        return p1 + p2 / x + p3 / x ** 2

    lb = [0, 0, 0]
    ub = [np.inf, np.inf, np.inf]
    options = {'maxiter': 1000, 'maxfev': 2000}

    # finding the parameters that best fit the data (AE):
    AE_fit_params, _ = curve_fit(F_model, subsample_size, average_information_versus_sample_size, bounds=(lb, ub), method='trf')
    AE_fitted_model = AE_fit_params[0] + AE_fit_params[1] / subsample_size + AE_fit_params[2] / subsample_size ** 2
    

    # extrapolating the information for each cell:
    if information_versus_sample_size.shape[0] > 1:
        N = information_versus_sample_size.shape[0]
        BAE_information = np.empty((N, 1))
        BAE_fit_R_2 = np.empty((N, 1))
        for n in range(N):
            this_information_versus_sample_size = information_versus_sample_size[n, :]
            if np.max(this_information_versus_sample_size) > 0:
                a_0 = this_information_versus_sample_size[-1]
                b_0 = this_information_versus_sample_size[0] - this_information_versus_sample_size[-1]
                c_0 = 1 / middle_sample_size * (this_information_versus_sample_size[0] - this_information_versus_sample_size[middle_index]) / (this_information_versus_sample_size[middle_index] - this_information_versus_sample_size[-1])

                initial_parameters = [a_0, b_0, c_0]

                def F_model(x, p1,p2,p3):
                    return p1 + p2 / (1 + p3 * x)

                lb = [0, 0, 0]
                ub = [np.inf, np.inf, np.inf]
                options = {'maxiter': 1000, 'maxfev': 2000}

                # finding the parameters that best fit the data:
                this_BAE_fit_params, _ = curve_fit(F_model, subsample_size, this_information_versus_sample_size, bounds=(lb, ub), method='trf')
                BAE_information[n] = this_BAE_fit_params[0]
                this_BAE_fitted_model = this_BAE_fit_params[0] + (this_BAE_fit_params[1]) / (1 + this_BAE_fit_params[2] * subsample_size)
                BAE_fit_R_2[n] = 1 - np.mean((this_BAE_fitted_model - this_information_versus_sample_size) ** 2) / np.var(this_information_versus_sample_size)
    else:
        BAE_information = average_BAE_information
        BAE_fit_R_2 = average_BAE_fit_R_2


    if plot_results or save_figures:
        if plot_results:
            plt.figure()
        else:
            plt.figure('Visible', 'off')
        plt.plot(subsample_size/subsample_size[-1], average_information_versus_sample_size, 'ob', linewidth=2)

        plt.plot(subsample_size/subsample_size[-1], AE_fitted_model, '-', color=[1, 0.5, 0], linewidth=2)
        plt.plot([0, 1], [AE_fit_params[0], AE_fit_params[0]], '--', color=[1, 0.5, 0], linewidth=2)
        plt.plot(subsample_size/subsample_size[-1], BAE_fitted_model, '-g', linewidth=2)
        plt.plot([0, 1], [average_BAE_fit_params[0], average_BAE_fit_params[0]], '--g', linewidth=2)
        plt.plot(subsample_size/subsample_size[-1], average_information_versus_sample_size, 'ob', linewidth=2)
        plt.xlim([0, 1])
        plt.ylim([0, np.ceil(average_information_versus_sample_size[0])])
        if units == 'bit':
            plt.ylim([0, 1.1*average_information_versus_sample_size[0]])
        plt.xlabel('Subsample fraction')
        if units == 'bit/spike' or units == 'bit/sec':
            plt.ylabel('SI (' + units + ')')
        else:
            plt.ylabel('MI (' + units + ')')
        plt.legend(['Naive', 'AE fitted model', 'AE estimation', 'BAE fitted model', 'BAE estimation'])
        plt.legend().set_visible(False)
        plt.gca().set_visible(False)
        plt.axis('off')
        plt.gca().set_frame_on(False)
        plt.gca().set_aspect('equal')
        if save_figures:
            if units == 'bit/spike':
                plt.savefig(os.path.join(figures_directory, 'BAE method - SI bit per spike.png'))
            elif units == 'bit/sec':
                plt.savefig(os.path.join(figures_directory, 'BAE method - SI bit per sec.png'))
            else:
                plt.savefig(os.path.join(figures_directory, 'BAE method - MI.png'))
        plt.show()

        if plot_results:
            plt.figure()
        else:
            plt.figure('Visible', 'off')
        plt.plot(information_versus_sample_size[:, -1], BAE_information, '.', markersize=15, color='g')

        plt.plot([0, 1.1*np.max(information_versus_sample_size[:, -1])], [0, 1.1*np.max(information_versus_sample_size[:, -1])], '--k', linewidth=2)
        plt.xlim([0, 1.1*np.max(information_versus_sample_size[:, -1])])
        plt.ylim([0, 1.1*np.max(information_versus_sample_size[:, -1])])
        plt.gca().set_aspect('equal')
        if units == 'bit/spike' or units == 'bit/sec':
            plt.xlabel('Naive SI (' + units + ')')
        else:
            plt.xlabel('Naive MI (' + units + ')')
        plt.ylabel('BAE estimation (' + units + ')')
        plt.gca().set_frame_on(False)
        if save_figures:
            if units == 'bit/spike':
                plt.savefig(os.path.join(figures_directory, 'BAE versus naive information - SI bit per spike.png'))
            elif units == 'bit/sec':
                plt.savefig(os.path.join(figures_directory, 'BAE versus naive information - SI bit per sec.png'))
            else:
                plt.savefig(os.path.join(figures_directory, 'BAE versus naive information - MI.png'))
        plt.show()

    return BAE_information,average_BAE_information,BAE_fit_R_2,average_BAE_fit_R_2



def perform_SSR(information_versus_sample_size, shuffle_information_versus_sample_size, subsample_size, units, plot_results, save_figures, figures_directory):
    '''
    This functions corrects the upward bias in the naive calculation of
    information for limited sample sizes using the scaled shuffle reduction (SSR) method.
    SSR is based on assuming a fixed bias ratio between the naive and shuffle information
    and using two different subsample sizes to find this ratio and subtract a the scaled shuffle information.
    The obtained results are compared against the shuffle reduction (SR) method.

    Inputs:
        1. information_versus_sample_size - Matrix of size TxN with the estimated
        information of each of N neurons as a function of T different sample sizes
        2. shuffle_information_versus_sample_size - Matrix of size TxN with the estimated
        bounds and 3 when using only the data
        3. subsample_size - Vector of T different sample sizes
        4. units - Either bit/spike, bit/sec, or bit
        5. plot_results - 1 for plotting and 0 for not
        6. save_figures - 1 for saving the figures and 0 for not
        7. figures_directory - Path for saving the figures

    Outputs:
        1. SSR_information - Vector of size N with the estimated
        information for each neuron.
        2. average_SSR_information - Single value with the estimated
        average information for the population.
        3. SSR_stability - Vector of size N with the SSR stability for each neuron.
        4. average_SSR_stability - Single value with the average SSR stability for the population.
    '''

    SSR_information_versus_sample_size = information_versus_sample_size - shuffle_information_versus_sample_size * (information_versus_sample_size[:, 0, np.newaxis] - information_versus_sample_size) / (shuffle_information_versus_sample_size[:, 0, np.newaxis] - shuffle_information_versus_sample_size)
    SSR_information = information_versus_sample_size[:, -1] - shuffle_information_versus_sample_size[:, -1] * (information_versus_sample_size[:, 0] - information_versus_sample_size[:, -1]) / (shuffle_information_versus_sample_size[:, 0] - shuffle_information_versus_sample_size[:, -1])
    SSR_stability = 1 - np.nanstd(SSR_information_versus_sample_size, axis=1, ddof=0) / SSR_information * subsample_size[1] / subsample_size[-1]

    if information_versus_sample_size.shape[0] > 1:
        average_information_versus_sample_size = np.nanmean(information_versus_sample_size, axis=0)
        average_shuffle_information_versus_sample_size = np.nanmean(shuffle_information_versus_sample_size, axis=0)
        average_information_short_duration = np.nanmean(information_versus_sample_size[:, 0])  # for subsample duration t1
        average_shuffle_information_short_duration = np.nanmean(shuffle_information_versus_sample_size[:, 0])
        average_SSR_information_versus_sample_size = average_information_versus_sample_size - average_shuffle_information_versus_sample_size * (average_information_short_duration - average_information_versus_sample_size) / (average_shuffle_information_short_duration - average_shuffle_information_versus_sample_size)
        average_SR_information_versus_sample_size = average_information_versus_sample_size - average_shuffle_information_versus_sample_size
        average_SSR_information = average_SSR_information_versus_sample_size[-1]
        average_SSR_stability = 1 - np.nanstd(average_SSR_information_versus_sample_size, ddof=0) / average_SSR_information * subsample_size[1] / subsample_size[-1]
    else:
        average_information_versus_sample_size = information_versus_sample_size
        average_shuffle_information_versus_sample_size = shuffle_information_versus_sample_size
        average_SR_information_versus_sample_size = average_information_versus_sample_size - average_shuffle_information_versus_sample_size
        average_SSR_information_versus_sample_size = SSR_information_versus_sample_size
        average_SSR_information = SSR_information
        average_SSR_stability = SSR_stability

    # plotting the average average results for the SSR method:
    if plot_results or save_figures:
        if plot_results:
            plt.figure()
        else:
            plt.figure(visible=False)
        plt.plot(np.arange(1, len(average_information_versus_sample_size) + 1) / len(average_information_versus_sample_size), average_information_versus_sample_size, color='b', linewidth=2)

        plt.plot(np.arange(1, len(average_information_versus_sample_size) + 1) / len(average_information_versus_sample_size), average_shuffle_information_versus_sample_size, color='k', linewidth=2)
        plt.plot(np.arange(1, len(average_information_versus_sample_size) + 1) / len(average_information_versus_sample_size), average_SR_information_versus_sample_size, '-r', linewidth=2)
        plt.plot(np.arange(1, len(average_information_versus_sample_size) + 1) / len(average_information_versus_sample_size), average_SSR_information_versus_sample_size, '-m', linewidth=2)
        plt.ylim([0, np.ceil(np.max(average_information_versus_sample_size))])
        plt.xlim([0, 1])
        if units == 'bit':
            plt.ylim([0, 1.1 * np.max(average_information_versus_sample_size)])
        plt.xlabel('Subsample fraction')
        if units == 'bit/spike' or units == 'bit/sec':
            plt.ylabel('SI (' + units + ')')
        else:
            plt.ylabel('MI (' + units + ')')
        plt.legend(['Naive', 'Shuffle', 'SR', 'SSR'])
        plt.legend().set_visible(False)
        # plt.gca().set_fontsize(16)
        plt.box(False)
        plt.gca().set_aspect('equal')
        if save_figures:
            if units == 'bit/spike':
                plt.savefig(os.path.join(figures_directory, 'SSR method - SI bit per spike.fig'))
                plt.savefig(os.path.join(figures_directory, 'SSR method - SI bit per spike.png'))
            elif units == 'bit/sec':
                plt.savefig(os.path.join(figures_directory, 'SSR method - SI bit per sec.fig'))
                plt.savefig(os.path.join(figures_directory, 'SSR method - SI bit per sec.png'))
            else:
                plt.savefig(os.path.join(figures_directory, 'SSR method - MI.fig'))
                plt.savefig(os.path.join(figures_directory, 'SSR method - MI.png'))
        # plotting the individual cells results for the SSR method:
        if plot_results:
            plt.figure()
        else:
            plt.figure(visible=False)
        plt.plot(information_versus_sample_size[:, -1], SSR_information, '.', markersize=15, color='m')

        plt.plot([0, 1.1 * np.max(information_versus_sample_size[:, -1])], [0, 1.1 * np.max(information_versus_sample_size[:, -1])], '--k', linewidth=2)
        plt.xlim([0, 1.1 * np.max(information_versus_sample_size[:, -1])])
        plt.ylim([0, 1.1 * np.max(information_versus_sample_size[:, -1])])
        plt.gca().set_aspect('equal')
        if units == 'bit/spike' or units == 'bit/sec':
            plt.xlabel('Naive SI (' + units + ')')
        else:
            plt.xlabel('Naive MI (' + units + ')')
        plt.ylabel('SSR estimation (' + units + ')')
        # plt.gca().set_fontsize(16)
        plt.box(False)
        if save_figures:
            if units == 'bit/spike':
                plt.savefig(os.path.join(figures_directory, 'SSR versus naive information - SI bit per spike.fig'))
                plt.savefig(os.path.join(figures_directory, 'SSR versus naive information - SI bit per spike.png'))
            elif units == 'bit/sec':
                plt.savefig(os.path.join(figures_directory, 'SSR versus naive information - SI bit per sec.fig'))
                plt.savefig(os.path.join(figures_directory, 'SSR versus naive information - SI bit per sec.png'))
            else:
                plt.savefig(os.path.join(figures_directory, 'SSR versus naive information - MI.fig'))
                plt.savefig(os.path.join(figures_directory, 'SSR versus naive information - MI.png'))
        plt.show()

    return SSR_information, average_SSR_information, SSR_stability, average_SSR_stability