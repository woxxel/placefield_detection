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


def build_struct_PC_results(n_cells, nbin, n_trials, N_f=2, n_steps=100):

    results = {}

    results["status"] = {
        "is_place_cell": {},
        "SNR": np.full(n_cells, np.NaN),
        "r_value": np.full(n_cells, np.NaN),
        "MI_value": np.full(n_cells, np.NaN),
        ## p-value? z-score? Isec, MI, uMI, etc?
    }

    for method in ["peak", "information", "stability", "bayesian"]:
        results["status"]["is_place_cell"][f"{method}_method"] = np.zeros(
            n_cells, dtype=bool
        )

    results["fields"] = {
        "parameter": {
            ## for each parameter
            "global": {},  ## n x N_f x 3
            "local": {},  ## n x N_f x n_trials x 3
        },
        ## p_x stored at predefined values?
        ## local stored as sparse matrix?
        "p_x": {
            "global": {},  ## n x N_f x 100
            "local": {},  ## n x N_f x n_trials x 100
        },
        ## x stored only once for each session?
        "x": {
            "global": {},  ## n x N_f x 100
            "local": {},  ## n x N_f x n_trials x 100
        },
        "logz": np.zeros((n_cells, N_f + 1, 2)),
        "active_trials": np.zeros((n_cells, N_f, n_trials)),
        ## need to be calculated extra
        "reliability": np.full((n_cells, N_f), np.NaN),
        "n_modes": np.zeros((n_cells, N_f), dtype=int),
    }

    key = "A0"
    results["fields"]["parameter"]["global"][key] = np.zeros((n_cells, 3))
    results["fields"]["p_x"]["global"][key] = np.zeros((n_cells, n_steps))
    results["fields"]["x"]["global"][key] = np.zeros((n_cells, n_steps))

    results["fields"]["parameter"]["local"][key] = np.zeros((n_cells, n_trials, 3))
    results["fields"]["p_x"]["local"][key] = np.zeros((n_cells, n_trials, n_steps))
    results["fields"]["x"]["local"][key] = np.zeros((n_cells, n_trials, n_steps))

    for key in ["theta", "A", "sigma"]:
        results["fields"]["parameter"]["global"][key] = np.zeros((n_cells, N_f, 3))
        results["fields"]["p_x"]["global"][key] = np.zeros((n_cells, N_f, n_steps))
        results["fields"]["x"]["global"][key] = np.zeros((n_cells, N_f, n_steps))

        results["fields"]["parameter"]["local"][key] = np.zeros(
            (n_cells, N_f, n_trials, 3)
        )
        results["fields"]["p_x"]["local"][key] = np.zeros(
            (n_cells, N_f, n_trials, n_steps)
        )
        results["fields"]["x"]["local"][key] = np.zeros(
            (n_cells, N_f, n_trials, n_steps)
        )

    results["firingstats"] = {
        "rate": np.full(n_cells, np.NaN),
        "map": np.full((n_cells, nbin), np.NaN),
        "CI": np.full((n_cells, 2, nbin), np.NaN),
        "trial_map": np.full((n_cells, n_trials, nbin), np.NaN),
    }

    ## if method is called for nCells = 1, collapse data from first dimension
    for field in results.keys():
        for key in results[field].keys():
            results[field][key] = np.squeeze(results[field][key])

            # results["status"] = {}

            # for key in [
            #     "MI_value",
            #     "MI_p_value",
            #     "MI_z_score",
            #     "Isec_value",
            #     "Isec_p_value",
            #     "Isec_z_score",
            #     "SNR",
            #     "r_value",
            # ]:
            #     results["status"][key] = np.full(nCells, np.NaN)

            # results["fields"] = {
            #     # "parameter": np.full(
            #     #     (nCells, 5, 4, nStats), np.NaN
            #     # ),  ### (mean,std,CI_low,CI_top)
            #     # "p_x": np.zeros((nCells, 5, nbin)),  ##sp.sparse.COO((nCells,3,nbin)),#
            #     # "reliability": np.zeros((nCells, 5)) * np.NaN,
            #     # "Bayes_factor": np.zeros((nCells, 5, 2)) * np.NaN,
            #     # "nModes": np.zeros(nCells).astype("int"),
            # }

            # results["firingstats"] = {
            #     "rate": np.full(nCells, np.NaN),
            #     "map": np.zeros((nCells, nbin)) * np.NaN,
            #     "std": np.zeros((nCells, nbin)) * np.NaN,
            #     "CI": np.zeros((nCells, 2, nbin)) * np.NaN,
            #     "trial_map": np.zeros((nCells, trial_ct, nbin)) * np.NaN,
            #     "trial_field": np.zeros((nCells, 5, trial_ct), "bool"),
            #     "parNoise": np.zeros((nCells, 2)) * np.NaN,
            # }

    return results
