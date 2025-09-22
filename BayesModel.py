import logging, time, os, warnings, signal
import numpy as np
from scipy.special import factorial as sp_factorial, erfinv, erf
from scipy.ndimage import gaussian_filter1d as gauss_filter
from scipy.interpolate import interp1d
from scipy.stats import chi2

from dataclasses import dataclass, fields as attributes

from .HierarchicalBayesModel import HierarchicalModel
from .HierarchicalBayesModel.structures import build_distr_structure_from_params, build_key, prior_structure, halfnorm_ppf, norm_ppf, bounded_flat, parse_name_and_indices

from .utils import circmean as weighted_circmean, model_of_tuning_curve
from .analyze_results import build_inference_results

os.environ["OMP_NUM_THREADS"] = "1"
logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings("ignore")

logger = logging.getLogger("ultranest")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)


class HierarchicalBayesInference(HierarchicalModel):    

    def prepare_data(self, event_counts, T, dimension_names=None, iter_dims=None):

        super().prepare_data(event_counts, T, dimension_names, iter_dims)
        # self.N = N[np.newaxis, ...]

        self.n_samples, self.n_bin = self.dimensions["shape"][-2:]

        self.x_arr = np.broadcast_to(np.arange(self.n_bin),(1,)*self.dimensions["n"]+(self.n_bin,))

        ## pre-calculate log-factorial for speedup
        self.log_N_factorial = np.log(sp_factorial(self.data["event_counts"]))

    def set_priors(self, priors_init=None, N_f=1):

        self.N_f = N_f
        if priors_init is None:

            firing_map = self.data["event_counts"].sum(axis=0) / self.data["T"].sum(axis=0)
            A0_guess, A_guess = np.maximum(np.percentile(
                firing_map, [10, 90]
            ), 0.1)

            self.n_bin

            self.priors_init = {}
            self.priors_init["A0"] = prior_structure(
                halfnorm_ppf,
                loc=0.0,
                scale=A0_guess,
                label="$A_0$",
            )

            for f in range(N_f):

                self.priors_init[build_key("A", "field", f)] = prior_structure(
                    halfnorm_ppf,
                    loc=0.0,
                    scale=A_guess,
                    label=f"$A_{f}$",
                )
                self.priors_init[build_key("sigma", "field", f)] = prior_structure(
                    halfnorm_ppf,
                    loc=0.5,
                    scale=self.n_bin / 10.0,
                    label=f"$\\sigma_{f}$",
                )
                self.priors_init[build_key("theta", "field", f)] = prior_structure(
                    norm_ppf,
                    mean=prior_structure(bounded_flat, low=0, high=self.n_bin),
                    sigma=prior_structure(halfnorm_ppf, loc=0, scale=self.n_bin / 20.0),
                    label=f"$\\theta_{f}$",
                    shape=(self.n_samples,)
                )

            # assert (
            #     A0_guess > 5
            # ), "Initial guess for A0is too high (above 5 Hz), which has been found to cause problems in the past (ultra-long run time). Execution is stopped to prevent this."

        else:
            self.priors_init = priors_init

        super().set_priors(self.priors_init)

    def set_logp_func(
        self, vectorized=True, penalties=["reliability", "overlap"]
    ):
        """
        TODO:
            instead of finding correlated trials before identification, run bayesian
            inference on all neurons, but adjust log-likelihood:

                * take placefield position, width, etc as hierarchical parameter
                    (narrow distribution for location and baseline activity?)
                * later, calculate logl for final parameter set to obtain active-trials (better logl)

            make sure all of this runs kinda fast!

            check, whether another hierarchy level should estimate noise-distribution parameters for overall data
                (thus, running inference only once on complete session, with N*4 parameters)
        """

        import matplotlib.pyplot as plt

        def get_logp(
            p_in, get_active_model=False, get_logp=False, get_tuning_curve=False
        ):

            t_ref = time.time()
            ## adjust shape of incoming parameters to vectorized analysis
            if len(p_in.shape) == 1:
                p_in = p_in[np.newaxis, :]
            N_in = p_in.shape[0]

            self.timeit()
            params = self.get_params_from_p(p_in)
            params = build_distr_structure_from_params(params, "field", place_field)

            # params = self.from_p_to_params(p_in)
            self.timeit("transforming parameters")

            tuning_curve_models = self.model_of_tuning_curve(params, stacked=True)
            if get_tuning_curve:
                return tuning_curve_models
            self.timeit("tuning curve model")

            logp_at_trial_and_position = np.zeros(
                (2**self.N_f, N_in, self.n_samples, self.n_bin)
            )
            logp_at_trial_and_position[0, ...] = self.probability_of_spike_observation(
                tuning_curve_models[0, ...]
            )

            if self.N_f > 0:
                if self.N_f > 1:
                    for field_model in range(self.N_f):
                        logp_at_trial_and_position[field_model + 1, ...] = (
                            self.probability_of_spike_observation(
                                tuning_curve_models[[0, field_model + 1], ...].sum(
                                    axis=0
                                )
                            )
                        )

                ## also, calculate log-likelihood for all fields combined
                logp_at_trial_and_position[-1, ...] = (
                    self.probability_of_spike_observation(
                        tuning_curve_models.sum(axis=0)
                    )
                )
                self.timeit("poisson")

                infield_range = self.generate_infield_ranges(params)
                self.timeit("infield ranges")

                AIC = self.compute_AIC(logp_at_trial_and_position, infield_range)
                self.timeit("AIC")

                active_model = self.obtain_active_model(AIC)
                if get_active_model:
                    return active_model
                self.timeit("active model")

            else:
                active_model = np.ones((1, N_in, self.n_samples), "bool")
                infield_range = None

            if get_logp:
                return logp_at_trial_and_position, active_model

            self.log.debug(
                (f"{logp_at_trial_and_position.shape} {logp_at_trial_and_position=}")
            )
            logp = np.sum(
                logp_at_trial_and_position,
                where=active_model[..., np.newaxis],
                axis=(3, 2, 0),
            )
            self.timeit("raw logp")

            self.log.debug((f"{logp=}"))

            if self.N_f > 0:
                logp -= self.calculate_logp_penalty(
                    p_in,
                    params,
                    logp_at_trial_and_position,
                    active_model,
                    infield_range,
                    penalties,
                )
                self.timeit("penalties")
            self.log.debug((f"{logp=}"))

            if vectorized:
                return logp
            else:
                return logp[0]

        return get_logp

    def generate_infield_ranges(self, params, cut_range=2.0):
        ## define ranges, in which the different models are compared
        N_in = params["A0"].shape[0]
        # nSamples = self.n_samples if len(self.hierarchical) > 0 else 1
        infield_range = np.zeros((self.N_f, N_in, self.n_samples, self.n_bin), dtype=bool)

        for field_model, field in enumerate(
            params["fields"]
        ):  # actually don't need the "if" before
            lower = np.floor(
                np.mod(field.theta - cut_range * field.sigma, self.n_bin)
            ).astype("int")
            upper = np.ceil(
                np.mod(field.theta + cut_range * field.sigma, self.n_bin)
            ).astype("int")

            for i in range(N_in):
                for trial in range(self.n_samples):
                    if lower[i, trial] < upper[i, trial]:
                        infield_range[
                            field_model, i, trial, lower[i, trial] : upper[i, trial]
                        ] = True
                    else:
                        infield_range[field_model, i, trial, lower[i, trial] :] = True
                        infield_range[field_model, i, trial, : upper[i, trial]] = True
        return infield_range

    def obtain_active_model(self, AIC):
        """
        something seems off with how area is defined and used...
        which axis should I apply reliability penalty to? how is it interpreted, if f(i,j) is the off-diagonal field? ...
        """
        ## entry 0 should be nofield model and all possible combinations of field models
        ## I'm actually not quite sure why it evolves to f**2, but it holds
        ## now, how do I properly assign IDs to to model (especially combinations)?

        N_in = AIC.shape[1]
        active_model_reference = np.argmin(AIC, axis=0)

        active_model = np.zeros((2**self.N_f, N_in, self.n_samples), dtype=bool)

        active_model[0, ...] = np.all(active_model_reference == 0, axis=1)
        if self.N_f == 1:
            active_model[1, ...] = np.any(active_model_reference == 1, axis=1)

        if self.N_f == 2:
            active_model[1, ...] = np.any(active_model_reference == 1, axis=1) & np.all(
                active_model_reference != 2, axis=1
            )
            active_model[2, ...] = np.any(active_model_reference == 2, axis=1) & np.all(
                active_model_reference != 1, axis=1
            )
            active_model[-1, ...] = np.all(~active_model[:-1, ...], axis=0)

        return active_model

    def compute_AIC(self, logp_at_trial_and_position, infield_range):
        N_in = logp_at_trial_and_position.shape[1]

        AIC = np.zeros((self.N_f + 1, N_in, self.N_f, self.n_samples))
        for field_area in range(self.N_f):

            nDatapoints = infield_range[field_area, ...].sum(axis=-1)

            ## calculate trial-wise log-likelihoods for both models
            logp_field_trials = np.sum(
                logp_at_trial_and_position,
                where=infield_range[[field_area], ...],
                axis=-1,
            )

            for field_model in range(self.N_f + 1):
                """
                consider trials to be place-coding, when Akaike information
                criterion (AIC) is lower than nofield-model. Number of parameters
                for each trial is 1 (no field) vs 4 (single field)
                """
                # off_field = (field_model>0) and (field_model != (field_area+1))
                nParameter = 1 + 3 * (field_model > 0)  # + 3*off_field

                AIC[field_model, :, field_area, :] = (
                    nParameter * np.log(nDatapoints)
                    - 2 * logp_field_trials[field_model, ...]
                )

        return AIC

    def calculate_logp_penalty(
        self,
        p_in,
        params,
        logp_at_trial_and_position,
        active_model,
        infield_range,
        penalties=["parameter_bias", "reliability", "overlap"],
        penalty_factor=None,
        no_go_factor=10**6,
    ):
        """
        calculates several penalty values for the log-likelihood to adjust inference
            - zeroing_penalty: penalizes parameters to be far from 0

            - centering_penalty: penalizes parameters to be far from meta-parameter

            - activation_penalty: penalizes trials to be place-coding
                    This could maybe introduce AIC as penalty factor, to bias towards non-coding, and only consider it to be better, when it surpasses AIC? (how to properly implement?)

        """

        if not penalty_factor:
            ## choose the penalty to be equal to the AIC difference between nofield and field model
            penalty_factor = 3.0 * np.log(self.n_bin)

        N_in = p_in.shape[0]

        if self.N_f > 1:
            active_model[1, active_model[-1, ...]] = True
            active_model[2, active_model[-1, ...]] = True

        zeroing_penalty = np.zeros(N_in)
        centering_penalty = np.zeros(N_in)

        if "parameter_bias" in penalties:
            ## for all non-field-trials, introduce "pull" towards 0 for all parameters to avoid flat posterior
            ### for all field-trials, enforce centering of parameters around active meta parameter

            for key in self.priors:

                _,f = parse_name_and_indices(key, ["field"])

                ## hierarchical parameters fluctuate around meta-parameters, and should be centered around them, as well as bias towards them for non-field trials
                if self.priors[key]["n"] > 1:

                    dParam_trial_from_total = (
                        p_in[
                            :,
                            self.priors[key]["idx"] : self.priors[key]["idx"]
                            + self.priors[key]["n"],
                        ]
                        - p_in[:, [self.priors[key]["idx_mean"]]]
                    ) / p_in[:, [self.priors[key]["idx_sigma"]]]

                    zeroing_penalty += penalty_factor * (
                        (dParam_trial_from_total * (~active_model[f, ...])) ** 2
                    ).sum(axis=1)
                    centering_penalty += (
                        penalty_factor
                        * ((dParam_trial_from_total * active_model[f, ...]).sum(axis=1))
                        ** 2
                    )
            self.timeit("zeroing/centering penalty")

        overlap_penalty = np.zeros(N_in)
        if ("overlap" in penalties) and (self.N_f > 1):
            overlap_range = np.all(infield_range, axis=0).sum(axis=-1)
            for field in params["fields"]:
                overlap_penalty += penalty_factor * norm_cdf(
                    overlap_range - 2 * field.sigma, 0, field.sigma
                ).sum(axis=-1)
            self.timeit("overlap penalty")

        reliability_penalty = np.zeros(N_in)
        if "reliability" in penalties:

            dlogp = np.zeros((self.N_f, N_in))
            logp_nofield = logp_at_trial_and_position[0, ...].sum(axis=-1)
            for f in range(self.N_f):
                logp_field = logp_at_trial_and_position[f + 1, ...].sum(axis=-1)
                dlogp_trials = np.maximum(logp_nofield - logp_field, 0)
                dlogp[f, :] = np.sum(
                    dlogp_trials,
                    where=~np.any(active_model[[f + 1, -1], ...], axis=0),
                    axis=-1,
                )
            # np.maximum(dlogp,0,out=dlogp)
            active_trials = active_model[range(1, self.N_f + 1), ...].sum(axis=-1)
            # assert np.all(dlogp>0), f'dlogp should be positive, {dlogp=}'

            # print(active_model[[1,2,3],...])

            reliability = active_trials / self.n_samples
            reliability_sigmoid = 1 - 1 / (1 + np.exp(-20 * (reliability - 0.3)))
            reliability_penalty = (reliability_sigmoid * dlogp).sum(axis=0)

            ## this is temporarily introduced - needs to be checked thoroughly!
            reliability_penalty += (np.maximum(4-active_trials,0)/3 *dlogp).sum(axis=0)

            # print(f"{reliability=}, {reliability_penalty=}")
            # print(f"{reliability*dlogp}")
            # reliability_penalty = penalty_factor * ((reliability>0) * (1. + (1.-reliability))).sum(axis=0)
            # reliability_penalty = penalty_factor * (1. + (1.-reliability)).sum(axis=0)
            self.timeit("reliability penalty")

        self.log.debug(("penalty (zeroing):", zeroing_penalty))
        self.log.debug(("penalty (centering):", centering_penalty))
        self.log.debug(("penalty (overlap):", overlap_penalty))
        self.log.debug(("penalty (reliability):", reliability_penalty))
        # self.log.debug(('penalty (activation):',activation_penalty))

        # self.log.debug(('dParams:',dParams_trial_from_total))

        ## introduce penalty for parameters to be below 0
        lower_bound_0_penalty = np.zeros(N_in)
        for field in params["fields"]:
            for attr in attributes(field):
                if np.any(getattr(field,attr.name) < 0):
                    lower_bound_0_penalty += no_go_factor * np.sum(
                        -getattr(field,attr.name), where=getattr(field,attr.name) < 0, axis=-1
                    )
        self.timeit("lower_bound_0 penalty")

        ordered_fields_penalty = np.zeros(N_in)
        if self.N_f > 1:
            ordered_fields_penalty = no_go_factor * np.maximum(
                0, params["fields"][0].theta - params["fields"][1].theta
            ).sum(axis=-1)
        self.timeit("ordered fields penalty")

        return (
            zeroing_penalty
            + centering_penalty
            + overlap_penalty
            + reliability_penalty
            + lower_bound_0_penalty
            + ordered_fields_penalty
        )
        # return zeroing_penalty, centering_penalty, activation_penalty

    def probability_of_spike_observation(self, nu):
        ## get probability to observe N spikes (amplitude) within dwelltime for each bin in each trial
        logp = (
            self.data["event_counts"] * np.log(nu * self.data["T"])
            - self.log_N_factorial
            - nu * self.data["T"]
        )

        logp[np.logical_and(nu == 0, self.data["event_counts"] == 0)] = 0
        logp[np.isnan(logp)] = -10.0
        logp[np.isinf(logp)] = -100.0  # np.finfo(logp.dtype).min
        return logp

    def model_of_tuning_curve(
        self,
        params,
        fields="all",
        stacked=False,
        # int | str | None
    ):

        return model_of_tuning_curve(
            self.x_arr, params, self.n_bin, self.n_samples, fields, stacked
        )

    # def from_p_to_params(self, p_in):
    #     """
    #     transform p_in to parameters for the model
    #     """
    #     params = {}

    #     if self.N_f > 0:
    #         params["fields"] = []
    #         for _ in range(self.N_f):
    #             params["fields"].append({})

    #     for key in self.priors:
    #         if self.priors[key]["meta"]:
    #             continue

    #         if key.startswith("fields"):
    #             nField, key_param = key.split("_")
    #             nField = int(nField[2:]) - 1
    #             params["fields"][nField][key_param] = p_in[
    #                 :,
    #                 self.priors[key]["idx"] : self.priors[key]["idx"]
    #                 + self.priors[key]["n"],
    #             ]
    #         else:
    #             key_param = key.split("__")[0]
    #             params[key_param] = p_in[
    #                 :,
    #                 self.priors[key]["idx"] : self.priors[key]["idx"]
    #                 + self.priors[key]["n"],
    #             ]

    #     return params

    # def from_results_to_params(self, results=None):
    #     """
    #     transform results to parameters for the model
    #     """
    #     results = results or self.inference_results
    #     params = {}

    #     if self.N_f > 0:
    #         params["fields"] = []
    #         for _ in range(self.N_f):
    #             params["fields"].append({})

    #     params["A0"] = results["fields"]["parameter"]["global"]["A0"][np.newaxis, 0]
    #     for key in ["theta", "A", "sigma"]:
    #         for f in range(self.N_f):
    #             if results["fields"]["parameter"]["local"][key] is None:
    #                 params["fields"][f][key] = results["fields"]["parameter"]["global"][
    #                     key
    #                 ][np.newaxis, f, 0]
    #             else:
    #                 params["fields"][f][key] = results["fields"]["parameter"]["local"][key][
    #                     np.newaxis, f, :, 0
    #                 ]

    #     return params

    def timeit(self, msg=None):
        if not msg is None:  # and (self.time_ref):
            self.log.debug(f"time for {msg}: {(time.time()-self.time_ref)*10**6}")

        self.time_ref = time.time()

    def model_comparison(
        self,
        hierarchical=["theta"],
        wrap=[],
        show_status=False,
        limit_execution_time=None,
    ):
        t_start = time.time()
        self.inference_results = build_inference_results(
            N_f=2,
            nbin=self.n_bin,
            mode="bayesian",
            n_trials=self.n_samples,
            hierarchical=hierarchical,
        )

        if limit_execution_time:

            def handler(signum, frame):
                print("Forever is over!")
                raise TimeoutException("end of time")

            signal.signal(signal.SIGALRM, handler)

        if (self.N > 0).sum() > 10:
            # if (activity[self.behavior["active"]] > 0).sum() < 10:
            # print("Not enough instances of activity detected")
            # return None

            previous_logz = -np.inf
            for f in range(2 + 1):
                if show_status:
                    print(f"\n{f=}\n")

                try:
                    if limit_execution_time:
                        signal.alarm(limit_execution_time)
                    self.set_priors(N_f=f, hierarchical_in=hierarchical, wrap=wrap)

                    sampling_results = self.run_sampling(
                        penalties=["overlap", "reliability"],
                        improvement_loops=2,
                        show_status=show_status,
                    )

                    if limit_execution_time:
                        signal.alarm(0)

                    self.inference_results["fields"]["logz"][f, 0] = sampling_results[
                        "logz"
                    ]
                    self.inference_results["fields"]["logz"][f, 1] = sampling_results[
                        "logzerr"
                    ]

                    ## 3 degrees of freedom, as statistic depends on difference of dof between models
                    if chi2.sf(-2*(previous_logz - sampling_results["logz"]), 3) > 0.01:
                        # if previous_logz > sampling_results["logz"]:
                        # print("chi statistic is not significant, stopping inference")
                        # print(
                        #     f"previous logz: {previous_logz:.2f}, current logz: {sampling_results['logz']:.2f}"
                        # )
                        # print("chi2:",chi2.sf(-2*(previous_logz - sampling_results["logz"]), 3))
                        self.N_f = f - 1
                        break

                    self.inference_results["fields"]["n_modes"] = self.N_f
                    self.store_inference_results(sampling_results)
                    previous_logz = sampling_results["logz"]
                except Exception as exc:
                    print("Exception:", exc)
                    break

            self.calculate_general_statistics()

            # if plot:
            #     self.display_results()

            string_out = f"Model comparison finished after {time.time() - t_start:.2f}s with evidences: "
            for f in range(2 + 1):
                if not np.isnan(self.inference_results["fields"]["logz"][f, 0]):
                    string_out += f"\t {f=} {'*' if f==self.inference_results['fields']['n_modes'] else ''}, logz={self.inference_results['fields']['logz'][f,0]:.2f}"
        else:
            self.calculate_general_statistics(which=["firingstats"])

            string_out = "Not enough instances of activity detected"

        print(string_out)

        return self.inference_results

    def calculate_general_statistics(self, which=["fields", "firingstats"]):

        if "fields" in which:
            # ## number of fields of best model
            # if np.all(np.isnan(self.inference_results["fields"]["logz"][:, 0])):
            #     field_model = -1
            # else:
            #     field_model = np.nanargmax(
            #         self.inference_results["fields"]["logz"][:, 0]
            #     )
            #     self.inference_results["fields"]["n_modes"] = field_model

            self.inference_results["is_place_cell"] = self.inference_results["fields"]["n_modes"] > 0

            ## reliability of place fields
            for f in range(self.inference_results["fields"]["n_modes"]):
                self.inference_results["fields"]["reliability"][f] = (
                    self.inference_results["fields"]["active_trials"][f, ...] > 0.5
                ).sum() / self.n_samples

        # if "firingstats" in which:
        #     ## firing rate statistics
        #     self.inference_results["firingstats"]["trial_map"] = self.N.sum(
        #         axis=0
        #     ) / self.data["T"].sum(axis=0)
        #     self.inference_results["firingstats"]["map"] = self.N.sum(
        #         axis=(0, 1)
        #     ) / self.data["T"].sum(axis=(0, 1))
        #     self.inference_results["firingstats"]["rate"] = (
        #         self.N.sum() / self.data["T"].sum()
        #     )

    # def run_sampling(
    #     self,
    #     prior_transform,
    #     loglikelihood,
    #     parameter_names,
    #     n_live=100,
    #     improvement_loops=2,
    #     show_status=False,
    # ):
    #     my_prior_transform = self.set_prior_transform(vectorized=True)
    #     penalties=["overlap", "reliability"],
    #     my_likelihood = self.set_logp_func(vectorized=True, penalties=penalties)

    #     ## setting up the sampler

    #     # ## nested sampling parameters
    #     # NS_parameters = {
    #     #     "min_num_live_points": n_live,
    #     #     "max_num_improvement_loops": improvement_loops,
    #     #     "max_iters": 50000,
    #     #     "cluster_num_live_points": 20,
    #     # }

    #     sampler = ultranest.ReactiveNestedSampler(
    #         self.paramNames,
    #         my_likelihood,
    #         my_prior_transform,
    #         wrapped_params=self.wrap,
    #         vectorized=True,
    #         num_bootstraps=20,
    #         ndraw_min=512,
    #     )

    #     sampling_result = None
    #     n_steps = 10  # hbm.f * 10
    #     while True:
    #         try:
    #             # sampler.stepsampler = PopulationSliceSampler(
    #             #     popsize=2**4,
    #             #     nsteps=n_steps,
    #             #     generate_direction=generate_region_oriented_direction,
    #             # )

    #             # sampling_result = sampler.run(
    #             #     **NS_parameters,
    #             #     region_class=RobustEllipsoidRegion,
    #             #     update_interval_volume_fraction=0.01,
    #             #     show_status=show_status,
    #             #     viz_callback=False,
    #             # )

    #             # self.store_inference_results(sampling_result)
    #             break
    #         except Exception as exc:
    #             if type(exc) == KeyboardInterrupt:
    #                 break
    #             if type(exc) == TimeoutException:
    #                 raise TimeoutException("Sampling took too long")
    #             n_steps *= 2
    #             print(f"increasing step size to {n_steps=}")
    #             if n_steps > 100:
    #                 break
    #     return sampling_result


class PlaceFieldInferenceResults:

    def __init__(self):
        ## these given from "dimension"?
        self.n_bin = 0
        self.n_trials = 0
        self.N_f = 0

        self.parameter_names = []
        self.inference_results = {}
        pass

    def store_local_parameters(self, f, posterior, key, key_results):

        for trial in range(self.n_trials):
            key_trial = f"{key_results}__{trial}"
            self.inference_results["fields"]["parameter"]["local"][key][f, trial, 0] = (
                posterior[key_trial]["mean"]
            )
            self.inference_results["fields"]["parameter"]["local"][key][
                f, trial, 1:
            ] = posterior[key_trial]["CI"][[0, -1]]

            self.inference_results["fields"]["p_x"]["local"][key][f, trial, :] = (
                posterior[key_trial]["p_x"]
            )
        pass

    def store_global_parameters(self, f, posterior, key, key_results):
        if key == "A0":
            self.inference_results["fields"]["parameter"]["global"][key][0] = posterior[
                key_results
            ]["mean"]
            self.inference_results["fields"]["parameter"]["global"][key][1:] = (
                posterior[key_results]["CI"][[0, -1]]
            )

            self.inference_results["fields"]["p_x"]["global"][key] = posterior[
                key_results
            ]["p_x"]
        else:
            self.inference_results["fields"]["parameter"]["global"][key][f, 0] = (
                posterior[key_results]["mean"]
            )
            self.inference_results["fields"]["parameter"]["global"][key][f, 1:] = (
                posterior[key_results]["CI"][[0, -1]]
            )

            self.inference_results["fields"]["p_x"]["global"][key][f, :] = posterior[
                key_results
            ]["p_x"]

    def store_parameters(self, f, posterior, key, key_results):

        if key in self.hierarchical:
            self.store_global_parameters(f, posterior, key, f"{key_results}__mean")
            self.store_local_parameters(f, posterior, key, key_results)
        else:
            self.store_global_parameters(f, posterior, key, key_results)
            # for store_keys in ["parameter", "p_x"]:
            #     self.inference_results["fields"][store_keys]["local"][key] = None

    def store_inference_results(self, results, n_steps=100):

        if not hasattr(self, "inference_results"):
            self.inference_results = build_inference_results(
                N_f=2,
                nbin=self.n_bin,
                mode="bayesian",
                n_trials=self.n_trials
                n_steps=n_steps,
                hierarchical=self.hierarchical,
            )
        posterior = self.build_posterior(results)

        for key in ["A0", "theta", "A", "sigma"]:

            if key == "A0":
                ## A0 is place f ield-independent parameter
                self.store_parameters(0, posterior, key, key)
                pass
            else:
                ## other parameters are place-field dependent
                for f in range(self.N_f):
                    key_prior = f"PF{f+1}_{key}"
                    self.store_parameters(f, posterior, key, key_prior)

        self.inference_results["fields"]["logz"][self.N_f, 0] = results["logz"]
        self.inference_results["fields"]["logz"][self.N_f, 1] = results["logzerr"]

        N_draws = 1000
        my_logp = self.set_logp_func(penalties=["overlap", "reliability"])

        active_model = my_logp(
            results["weighted_samples"]["points"][-N_draws:, :],
            get_active_model=True,
        )

        if self.N_f > 1:
            active_model[1, active_model[-1, ...]] = True
            active_model[2, active_model[-1, ...]] = True

        for f in range(self.N_f):
            self.inference_results["fields"]["active_trials"][f, ...] = active_model[
                f + 1, ...
            ].sum(axis=0)
        self.inference_results["fields"]["active_trials"] /= N_draws

    def build_posterior(self, results, smooth_sigma=1, n_steps=100, use_dynesty=False):

        posterior = {}

        self.posterior_arrays = {
            "A0": np.linspace(0, 2, n_steps + 1),
            "A": np.linspace(0, 50, n_steps + 1),
            "sigma": np.linspace(0, self.n_bin / 2.0, n_steps + 1),
            "theta": np.linspace(0, self.n_bin, n_steps + 1),
        }

        for i, key in enumerate(self.parameter_names):

            try:
                key_root, key_stat = key.split("__")
            except:
                key_root = key
                key_stat = None
            paramName = key_root.split("_")[-1]

            if use_dynesty:
                samp = results.samples[:, i]
                weights = results.importance_weights()

            else:
                samp = results["weighted_samples"]["points"][:, i]
                weights = results["weighted_samples"]["weights"]

            mean = (samp * weights).sum()

            qs = [0.001, 0.05, 0.341, 0.5, 0.841, 0.95, 0.999]

            if (paramName == "theta") and (key_stat != "sigma"):
                # print("wrap theta")
                mean = weighted_circmean(samp, weights=weights, low=0, high=self.n_bin)
                shift_from_center = mean - self.n_bin / 2.0

                samp[samp < np.mod(shift_from_center, self.n_bin)] += self.n_bin

            idx_sorted = np.argsort(samp)
            samples_sorted = samp[idx_sorted]

            # get corresponding weights
            sw = weights[idx_sorted]

            cumsw = np.cumsum(sw)
            quants = np.interp(qs, cumsw, samples_sorted)

            # nsteps = 101
            # low, high = quants[[0, -1]]
            # x = np.linspace(low, high, nsteps)
            x = self.posterior_arrays[paramName]

            if paramName == "theta":
                f = interp1d(
                    np.mod(samples_sorted, self.n_bin),
                    cumsw,
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                # import matplotlib.pyplot as plt

                # plt.figure()
                # plt.plot(x[:-1], f(x[1:]) - f(x[:-1]))
                # plt.show()
            else:
                f = interp1d(
                    samples_sorted, cumsw, bounds_error=False, fill_value="extrapolate"
                )

            posterior[key] = {
                "CI": np.mod(quants[1:-1], self.n_bin),
                "mean": mean,
                "p_x": (
                    ## cdf makes jump at wrap-point, resulting in a single negative value. "Maximum" fixes this, but is not ideal
                    np.maximum(
                        0,
                        (
                            gauss_filter(f(x[1:]) - f(x[:-1]), smooth_sigma)
                            if smooth_sigma > 0
                            else f(x[1:]) - f(x[:-1])
                        ),
                    )
                ),
            }

        return posterior


def norm_cdf(x, mu, sigma):
    return 0.5 * (1.0 + erf((x - mu) / (np.sqrt(2) * sigma)))


@dataclass
class place_field:
    A: float
    sigma: float
    theta: float


class TimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        pass
