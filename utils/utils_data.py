""" contains several functions, defining data and parameters used for analysis

  set_para

"""

import os
import numpy as np


class detection_parameters:

    nbin = 40
    qtl_steps = 4
    # coarse_factor = int(nbin / 20)

    # parameter = {}
    parameter_default = {
        "behavior": {
            "nbin": nbin,  # number of bins into which the location is comparted
            "bin_array": np.linspace(0, nbin - 1, nbin),
            "bin_array_centers": np.linspace(0, nbin, nbin + 1) - 0.5,
            "L_track": 120,
        },
        "activity_processing": {
            "gauss_filter_sigma": None,
            "baseline_percentile": 10,
        },
        "placefield_detection": {
            "f": 15,  # the frequency of the data
            "model_parameter_names": ["A_0", "A", "sigma", "theta"],
            "minimum_active_trial_fraction": 0.3,  # not a hard threshold, but introduces penalty around, especially below this value
            "posterior_percentiles_to_store": [0.025, 0.05, 0.95, 0.975],
            # "minimum_active_trial_number": 3,
        },
        "neuron_detection": {
            "SNR_thr": 2,
            "r_value_thr": 0.5,
            # "pxtomu": 536 / 512,
        },
        "shuffling": {
            "do_shuffle": True,
            "mode": "shuffle_trials",  ## how to shuffle: 'shuffle_trials', 'shuffle_global', 'randomize'
            "N_bs": 10000,
            "repnum": 1000,
        },
        "information_content": {
            "do_information": True,
            "which": "unbiased",  # change to unbiased estimator
            # "coarse_factor": coarse_factor,
            # "nbin_coarse": int(nbin / coarse_factor),
            "qtl_steps": 4,
            # "qtl_weight": np.ones(qtl_steps) / qtl_steps,	# should be set as setter
        },
        # "rate_thr": 4,
        # "width_thr": 5,
        # "sigma": 5,
        # "Ca_thr": 0,
        # 't_measures': get_t_measures(mouse),
        # "nP": nP,
    }

    def __init__(self, **kwargs):

        for key in self.parameter_default.keys():
            for sub_key in self.parameter_default[key].keys():

                setattr(
                    self,
                    sub_key,
                    (
                        kwargs[sub_key]
                        if sub_key in kwargs
                        else self.parameter_default[key][sub_key]
                    ),
                )

                # def set_paths(self, pathData, pathResults, suffix=""):

                #     self.params = self.params | {
                #         "pathData": pathData,
                #         "pathResults": pathResults,
                #         "pathFigures": os.path.join(
                #             pathResults, f"figures"
                #         ),  #'/home/wollex/Data/Science/PhD/Thesis/pics/Methods',
                #         ### provide names for distinct result files (needed?)
                #         "pathResults": os.path.join(pathResults, "placefields{suffix}.pkl"),
                #         # 'pathResults_status':       os.path.join(pathResults,'PC_fields%s_status.pkl'%suffix),
                #         # 'pathResults_fields':       os.path.join(pathResults,'PC_fields%s_para.pkl'%suffix),
                #         # 'pathResults_firingstats':  os.path.join(pathResults,'PC_fields%s_firingstats.pkl'%suffix),
                #     }


## -----------------------------------------------------------------------------------------------------------------------

# if nargin == 3:

# para.t_s = get_t_measures(mouse);
# para.nSes = length(para.t_s);

# time_real = false;
# if time_real
# t_measures = get_t_measures(mouse);
# t_mask_m = false(1,t_measures(nSes));
# for s = 1:nSes-1
# for sm = s+1:nSes
# dt = t_measures(sm)-t_measures(s);
# t_mask_m(dt) = true;
# end
# end
# t_data_m = find(t_mask_m);
# t_ses = t_measures;
# t_mask = t_mask_m;
# t_data = t_data_m;
# nT = length(t_data);
# else
# t_measures = get_t_measures(mouse);
# t_measures = t_measures(s_offset:s_offset+nSes-1);
##      t_measures = 1:nSes;    ## remove!
##      t_measures
# t_ses = linspace(1,nSes,nSes);
# t_data = linspace(1,nSes,nSes);
# t_mask = true(nSes,1);
# nT = nSes;
# end

# return para

