import numpy as np
from .HierarchicalBayesModel.NestedSamplingMethods import get_single_posterior_from_samples, get_samples_from_results

from .HierarchicalBayesModel.structures import parse_name_and_indices

"""
TODO:
* structure should be checked again: should "build results" be separate from class? or belong to it?

"""
class PlaceFieldInferenceResults:

    def __init__(self,**kwargs):
        ## these given from "dimension"?
        self.n_bin = kwargs.get("n_bin", 40)
        self.n_trials = kwargs.get("n_trials", 1)
        self.n_cells = kwargs.get("n_cells",1)

        # print(f"initializing PlaceFieldInferenceResults for {self.n_cells} cells, {self.n_bin} bins, {self.n_trials} trials")
        # if HBI is None:
        #     self.hierarchical = kwargs.get("hierarchical", [])
        # else:
        #     # HBI.set_priors(N_f=self.N_f)
        

        #     # self.priors = HBI.priors
        #     self.set_logp_func = HBI.set_logp_func

        #     self.periodic = HBI.periodic
            

        # self.build_results()


    def build_results(self, priors, n_steps=100, posterior_arrays=None
    ):
        # self.N_f = N_f#kwargs.get("N_f", 1 if HBI is None else HBI.N_f)

        self.hierarchical = []
        self.N_f = 0
        for key in priors:
            param_name, fields = parse_name_and_indices(key, ["field"])

            self.N_f = max(self.N_f, fields[0] + 1 if fields[0] is not None else 0)
            if priors[key]["n"] > 1:
                self.hierarchical.append(param_name)

        ## build dictionary shape
        base_dims = (self.n_cells,) if self.n_cells > 1 else ()
        self.fields = {
            "n_modes": np.full(base_dims,self.N_f, dtype=int),
            "parameter": {
                ## for each parameter
                "global": {},  ## n x N_f x 3
                "local": {},  ## n x N_f x n_trials x 3
            },
            ## local stored as sparse matrix?
            "p_x": {
                "global": {},  ## n x N_f x 100
                "local": {},  ## n x N_f x n_trials x 100
            },
            "x": {},
            "logz": np.full(base_dims + (2,), np.NaN),
            "active_trials": np.zeros(base_dims + (self.N_f, self.n_trials)),
            ## need to be calculated extra
            "reliability": np.full(base_dims + (self.N_f,), np.NaN),
        }

        ## define ranges for each parameter
        if posterior_arrays is None:
            self.set_posterior_arrays(n_steps=n_steps)
            self.fields["x"] = self.posterior_arrays
        else:
            self.fields["x"] = posterior_arrays
        

        ## fill dictionary with default values
        for key in priors.keys():
            param_name, indices = parse_name_and_indices(key, ["field", ""])
            field = indices[0]

            # if param_name in self.fields["parameter"]["local"]:
            #     continue  ## already created
    
            n = n_steps  # len(fields["x"][param_name]) - 1
            base_dims = (self.n_cells,) if self.n_cells > 1 else ()
            

            if field is not None:# or (not self.from_HBI and param_name in self.hierarchical):
                base_dims += (self.N_f,)
            
            # print(f"building storage for {key}, {param_name}, field={field}, base_dims={base_dims}, n={n}")
            
            if param_name in self.hierarchical:
                base_dims += (self.n_trials,)
                self.fields["parameter"]["local"][param_name] = np.full(base_dims + (3,), np.nan)
                self.fields["p_x"]["local"][param_name] = np.full(base_dims + (n,), np.nan)
            else:
                self.fields["parameter"]["global"][param_name] = np.full(base_dims + (3,), np.nan)
                self.fields["p_x"]["global"][param_name] = np.full(base_dims + (n,), np.nan)

            # print(f"done building storage for {key}, {param_name}, {self.fields['parameter']['global'].get(param_name, None)},")


        ## finally, map "theta" to "theta_mean"
        if "theta_mean" in self.fields["parameter"]["global"]:
            self.fields["parameter"]["global"]["theta"] = self.fields["parameter"]["global"]["theta_mean"]
            self.fields["p_x"]["global"]["theta"] = self.fields["p_x"]["global"]["theta_mean"]


    def store_inference_results(self, results, parameter_names, periodic=None, logp=None, mode="dynesty"):

        periodic = periodic if periodic is not None else [False]*len(parameter_names)
        
        self.samples = get_samples_from_results(results, mode=mode)
        
        self.fields["logz"][0] = results["logz"][-1]
        self.fields["logz"][1] = results["logzerr"][-1]

        for key_idx, key in enumerate(parameter_names):
            param_name, (field, trial) = parse_name_and_indices(key, ["field",""])

            # key_idx = parameter_names.index(key)
            # print(f"storing {key}, {param_name}, field={trials}")
            # if self.priors[param_name]["n"] > 1:
            if trial is not None:
                self.store_local_parameters(key, key_idx, periodic=periodic[key_idx])
            else:
                self.store_global_parameters(key, key_idx, periodic=periodic[key_idx])

        if logp is not None:
            self.calculate_active_trials(logp)
            self.calculate_field_reliability()


    def store_local_parameters(self, key, key_idx, periodic=False):

        param_name,(field,trial) = parse_name_and_indices(key, ["field",""])
        posterior = self.build_posterior_single(key, key_idx, periodic=periodic)
        # print(f"local, {param_name}, {key}, {field}, {trial}")

        self.fields["parameter"]["local"][param_name][field, trial, 0] = posterior["mean"]
        self.fields["parameter"]["local"][param_name][field, trial, 1:] = posterior["CI"][[0, -1]]

        self.fields["p_x"]["local"][param_name][field, trial, :] = posterior["p_x"]


    def store_global_parameters(self, key, key_idx, periodic=False):

        param_name,(field,trial) = parse_name_and_indices(key, ["field",""])
        posterior = self.build_posterior_single(key, key_idx, periodic=periodic)
        # print(f"global, {param_name}, {key}, {field}")

        if field is None:
            self.fields["parameter"]["global"][param_name][0] = posterior["mean"]
            self.fields["parameter"]["global"][param_name][1:] = posterior["CI"][[0, -1]]

            # print("p_x:",posterior[key_results]["p_x"])
            self.fields["p_x"]["global"][param_name] = posterior["p_x"]
        else:
            self.fields["parameter"]["global"][param_name][field, 0] = posterior["mean"]
            self.fields["parameter"]["global"][param_name][field, 1:] = posterior["CI"][[0, -1]]

            self.fields["p_x"]["global"][param_name][field, :] = posterior["p_x"]


    def calculate_active_trials(self, logp, cum_posterior_level=0.05):
        # my_logp = self.set_logp_func(penalties=["overlap", "reliability"])

        # for active trial calculation, take into account the samples part of the posterior that contribute 95% of the probability mass (could sort weights before doing so, but should give same or similar results in all cases)
        considered_idx = np.where(self.samples["weights"].cumsum() > cum_posterior_level)[0]
        N_draws = len(considered_idx)
        active_model = logp(
            self.samples["samples"][considered_idx[0]:, :],
            get_active_model=True,
        )

        if self.N_f > 1:
            active_model[1, active_model[-1, ...]] = True
            active_model[2, active_model[-1, ...]] = True

        for f in range(self.N_f):
            self.fields["active_trials"][f, ...] = active_model[
                f + 1, ...
            ].sum(axis=0)
        self.fields["active_trials"] /= N_draws


    def calculate_field_reliability(self):
        
        self.fields["reliability"] = (
            self.fields["active_trials"].sum(axis=1)
            / self.n_trials
        )


    def set_posterior_arrays(self, n_steps=100):
        self.posterior_arrays = {
            "A0": np.linspace(0, 2, n_steps + 1),
        }
        if self.N_f > 0:
            self.posterior_arrays.update({
                "A": np.linspace(0, 50, n_steps + 1),
                "sigma": np.linspace(0, self.n_bin / 2.0, n_steps + 1),
                "theta": np.linspace(0, self.n_bin, n_steps + 1),
            })
            self.posterior_arrays["theta_mean"] = self.posterior_arrays["theta"]
            self.posterior_arrays["theta_sigma"] = self.posterior_arrays["sigma"]

    def build_posterior_single(self, key, key_idx, periodic=False):

        param_name, _ = parse_name_and_indices(key, ["field",""])
        # i = self.parameter_names_all.index(key)

        return get_single_posterior_from_samples(self.samples["samples"][:,key_idx],self.samples["weights"],periodic=periodic,x=self.posterior_arrays[param_name])


def build_results(n_cells=1, n_bin=40, n_trials=None, modes=[], **kwargs):

    """
    This should be out of the structure for place fields, as it builds a general dictionary for results
    """
    results = {}
    # results["status"] = {
    #     "SNR": np.full(n_cells, np.NaN),
    #     "r_value": np.full(n_cells, np.NaN),
    #     # "MI_value": np.full(n_cells, np.NaN),
    #     ## p-value? z-score? Isec, MI, uMI, etc?
    # }

    base_dims = (n_cells,) if n_cells > 1 else ()

    results["firingstats"] = {
        "firing_rate": np.full(base_dims, np.NaN),
        "map_rates": np.full(base_dims + (n_bin,), np.NaN),
        "map_trial_rates": np.full(base_dims + (n_trials, n_bin), np.NaN),
    }

    # results = results if n_cells > 1 else squeeze_deep_dict(results, ax=0)

    for mode in modes:
        results[mode] = build_mode_results(
            n_cells=n_cells,
            mode=mode,
            n_trials=n_trials,
            n_bin=n_bin,
            **kwargs,
        )
    return results



def build_mode_results(
    n_cells=1, mode="bayesian", **kwargs
):

    results = {}

    results["is_place_cell"] = np.zeros(n_cells, dtype=bool)
    results["p_value"] = np.full(n_cells, np.NaN)

    if mode == "bayesian":
        # assert (HBI := kwargs.get("HBI", None)) is not None, "HBI model must be provided for Bayesian inference results."
        results["field_models"] = {}
        if (HBI := kwargs.get("HBI", None)) is not None:
            for n in range(kwargs.get("N_f",2)+1):
                HBI.set_priors(N_f=n)
                PFI = PlaceFieldInferenceResults(n_cells=n_cells, **kwargs)
                PFI.build_results(priors=HBI.priors)
                results["field_models"][n] = PFI.fields

            PFI.build_results(priors=HBI.priors)
            results["fields"] = PFI.fields
                
        # PFI = PlaceFieldInferenceResults(n_cells=n_cells,**kwargs)
        # results["fields"] = PFI.fields

    if mode == "threshold":
        results["fields"] = build_inference_results__thresholding(n_cells,N_f=kwargs.get("N_f",0))

    ## if method is called for nCells = 1, collapse data from first dimension
    # return results
    # if n_cells == 1:
    #     results = squeeze_deep_dict(results, ax=0)
    return results #if n_cells > 1 else squeeze_deep_dict(results, ax=0)


def build_inference_results__thresholding(n_cells=1, N_f=0):

    fields = {
        "n_modes": np.zeros(n_cells, dtype=int),
        "parameter": {
            "baseline": np.full(n_cells,np.NaN),
            "amplitude": np.full((n_cells, N_f),np.NaN),
            "location": np.full((n_cells, N_f),np.NaN),
            "width": np.full((n_cells, N_f),np.NaN),
        },
    }

    return fields



def handover_inference_results(
    results_source, results_target, idx=None, excluded_keys=["x"]
):
    """
    writing single-index entry to dictionary, honoring the overall structure
    """

    for key in results_source.keys():
        if key in excluded_keys:
            continue
        
        if isinstance(results_source[key], dict):
            # print(f"descending into {key}")
            # print(results_target.get(key))
            results_target[key] = handover_inference_results(
                results_source[key], results_target[key], idx
            )
        else:
            if (results_target.get(key) is not None) and (results_source.get(key) is not None):
                # print(f"entries {key}:",np.atleast_1d(results_source.get(key)).shape, np.atleast_1d(results_target.get(key)[idx,...]).shape)
                if np.all(np.atleast_1d(results_source.get(key)).shape==np.atleast_1d(results_target.get(key)[idx,...]).shape):
                    # print(f"{key} into fitting array")
                    results_target[key][idx, ...] = results_source[key]
                else:
                    # print(f"{key} into larger array")
                    # build slices according to dimensions of results_source
                    slices = tuple(slice(0, dim) for dim in np.atleast_1d(results_source[key]).shape)
                    results_target[key][(idx,) + slices] = results_source[key]
                    

    return results_target


def extract_inference_results(
    results_source, idx, results_target=None, excluded_keys=["x"]
):
    """
    extracting a single-index entry from dictionary, maintaining the overall structure
    """

    if results_target is None:
        results_target = {}

    for key in results_source.keys():

        if key == "x":
            results_target[key] = results_source[key]
        
        if key in excluded_keys:
            continue
        # print(f"extracting {key}...")
        
        if isinstance(results_source[key], dict):
            # if results_target.get(key, None) is None:
            #     results_target[key] = {}
            # print(f"descending into {key}")
            results_target[key] = extract_inference_results(
                results_source[key], idx, results_target.get(key)
            )
        else:
            # if not (results_source.get(key,None) is None) and not (results_target.get(key,None) is None):
            # print(f"entries {key}:",results_source[key][idx,...], results_target.get(key))
            results_target[key] = results_source[key][idx, ...]
            # print(f"extracted {key}:", results_target[key])
    # print("all done here")

    return results_target


def squeeze_deep_dict(d, ax=None):
    for key in d.keys():
        if isinstance(d[key], dict):
            d[key] = squeeze_deep_dict(d[key], ax)
        else:
            if isinstance(d[key], np.ndarray) and (d[key].shape[0] == 1):
                d[key] = np.squeeze(d[key], axis=ax)
    return d

