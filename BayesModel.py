import logging, time, os, warnings, signal
import numpy as np
from scipy.special import factorial as sp_factorial, erfinv, erf

from scipy.stats import chi2

from dataclasses import dataclass, fields as attributes

from .HierarchicalBayesModel import HierarchicalModel
from .HierarchicalBayesModel.structures import build_distr_structure_from_params, build_key, prior_structure, halfnorm_ppf, norm_ppf, bounded_flat, parse_name_and_indices
from .HierarchicalBayesModel.NestedSamplingMethods import run_sampling

from .utils import model_of_tuning_curve
# from .result_structures import build_inference_results

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
                    mean=prior_structure(bounded_flat, low=0, high=self.n_bin,periodic=True),
                    sigma=prior_structure(halfnorm_ppf, loc=0, scale=self.n_bin / 20.0),
                    label=f"$\\theta_{f}$",
                    shape=(self.n_samples,),
                    periodic=True,
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

        def get_logp(
            p_in, get_active_model=False, get_logp=False, get_tuning_curve=False
        ):

            ## adjust shape of incoming parameters to vectorized analysis
            if len(p_in.shape) == 1:
                p_in = p_in[np.newaxis, :]
            N_in = p_in.shape[0]

            self.timeit()
            params = self.get_params_from_p(p_in)
            params = build_distr_structure_from_params(params, "field", place_field)

            # print(params)
            # print(self.data)
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

                # AIC = self.compute_AIC(logp_at_trial_and_position, infield_range)
                # self.timeit("AIC")

                active_model = self.obtain_active_model(logp_at_trial_and_position, infield_range)
                if get_active_model:
                    return active_model
                self.timeit("active model")

            else:
                active_model = np.ones((1, N_in, self.n_samples), "bool")
                infield_range = None

            if get_logp:
                return logp_at_trial_and_position, active_model

            # self.log.debug(
            #     (f"{logp_at_trial_and_position.shape} {logp_at_trial_and_position=}")
            # )
            logp = np.sum(
                logp_at_trial_and_position,
                where=active_model[..., np.newaxis],
                axis=(3, 2, 0),
            )
            self.timeit("raw logp")

            # self.log.debug((f"{logp=}"))

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
            # self.log.debug((f"{logp=}"))

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

    def obtain_active_model(self, logp_at_trial_and_position, infield_range):
        """
        something seems off with how area is defined and used...
        which axis should I apply reliability penalty to? how is it interpreted, if f(i,j) is the off-diagonal field? ...
        """
        ## entry 0 should be nofield model and all possible combinations of field models
        ## I'm actually not quite sure why it evolves to f**2, but it holds
        ## now, how do I properly assign IDs to to model (especially combinations)?
        AIC = self.compute_AIC(logp_at_trial_and_position, infield_range)

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

        ### this also works for a single field
        # active_model[active_model_reference[0,0,:],:,range(self.n_samples)] = True

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
        penalties=["reliability", "overlap"],
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

        if "parameter_bias" in penalties:
            zeroing_penalty, centering_penalty = self.calculate_penalty_parameterbias(p_in, active_model, penalty_factor)
        else:
            zeroing_penalty = np.zeros(N_in)
            centering_penalty = np.zeros(N_in)

        if ("overlap" in penalties):
            overlap_penalty = self.calculate_penalty_overlap(params, infield_range, penalty_factor=penalty_factor)
        else:
            overlap_penalty = np.zeros(N_in)

        # print("calculating reliability penalty")
        if "reliability" in penalties:
            reliability_penalty = self.calculate_penalty_reliability(logp_at_trial_and_position, active_model)
        else:
            reliability_penalty = np.zeros(N_in)
        # print(f"reliability_penalty: {reliability_penalty}")
        
        lower_bound_0_penalty = self.calculate_penalty_nonegatives(params, no_go_factor)
        ordered_fields_penalty = self.calculate_penalty_ordered_fields(params, no_go_factor)

        # self.log.debug(("penalty (zeroing):", zeroing_penalty))
        # self.log.debug(("penalty (centering):", centering_penalty))
        # self.log.debug(("penalty (overlap):", overlap_penalty))
        # self.log.debug(("penalty (reliability):", reliability_penalty))
        # self.log.debug(('penalty (activation):',activation_penalty))

        # self.log.debug(('dParams:',dParams_trial_from_total))
        return (
            zeroing_penalty
            + centering_penalty
            + overlap_penalty
            + reliability_penalty
            + lower_bound_0_penalty
            + ordered_fields_penalty
        )

    
    def calculate_penalty_parameterbias(self, p_in, active_model, penalty_factor):
        """ 
        zeroing:    for all non-field-trials, introduce "pull" towards 0 for all parameters to avoid flat posterior
        centering:  for all field-trials, enforce centering of parameters around active meta parameter
        """
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
        return zeroing_penalty, centering_penalty


    def calculate_penalty_overlap(self, params, infield_range, penalty_factor=1.0):
        """
            penalizes fields that overlap too much (i.e. are not distinct enough)
            overlap is calculated as the range of positions, in which all fields are active
            (thus, if 2 fields are completely within each other, the penalty is maximal)

            penalty is independent of current logp, as heavy improvements should still be allowed
        """

        N_in = params["A0"].shape[0]
        penalty_offset = norm_cdf(-2,0,1)

        overlap_penalty = np.zeros(N_in)
        if self.N_f > 1:
            overlap_range = np.all(infield_range, axis=0).sum(axis=-1).mean()
            # print(f"overlap_range: {overlap_range}")
            for field in params["fields"]:
                # print(f"field:", field)
                overlap_penalty += penalty_factor * (norm_cdf(
                    overlap_range - 2 * field.sigma, 0, field.sigma
                ) - penalty_offset).sum(axis=-1)
        # print(f"overlap_penalty:",overlap_penalty)
        self.timeit("overlap penalty")
        return overlap_penalty


    def calculate_penalty_nonegatives(self, params, no_go_factor=10**6):
        
        ## introduce penalty for parameters to be below 0
        N_in = params["A0"].shape[0]
        lower_bound_0_penalty = np.zeros(N_in)
        for field in params["fields"]:
            for attr in attributes(field):
                if np.any(getattr(field,attr.name) < 0):
                    lower_bound_0_penalty += no_go_factor * np.sum(
                        -getattr(field,attr.name), where=getattr(field,attr.name) < 0, axis=-1
                    )
        self.timeit("lower_bound_0 penalty")
        return lower_bound_0_penalty

    def calculate_penalty_ordered_fields(self, params, no_go_factor=10**6):
        N_in = params["A0"].shape[0]
        ordered_fields_penalty = np.zeros(N_in)
        if self.N_f > 1:
            ordered_fields_penalty = no_go_factor * np.maximum(
                0, params["fields"][0].theta - params["fields"][1].theta
            ).sum(axis=-1)
        self.timeit("ordered fields penalty")
        return ordered_fields_penalty
    
        # return zeroing_penalty, centering_penalty, activation_penalty

    def calculate_penalty_reliability(self,
            logp_at_trial_and_position, active_model,
            threshold_active_trials = 5
        ):
        """
            penalizes fields that are not reliable enough (i.e. active in too few trials by rate or absolute numbers)
            dlogp is used as penalty baseline, effectively setting non-reliable field-logp to the value of no-field logp
        """
        N_in = active_model.shape[1]
        dlogp = np.zeros((self.N_f,N_in))
        logp_nofield = logp_at_trial_and_position[0, ...].sum(axis=-1)
        for f in range(self.N_f):
            logp_field = logp_at_trial_and_position[f + 1, ...].sum(axis=-1)

            dlogp_trials = np.maximum(logp_field - logp_nofield, 0)
            # dlogp[f,:] = dlogp_trials.sum()
            dlogp[f, :] = np.sum(
                dlogp_trials,
                where=np.any(active_model[[f + 1, -1], ...], axis=0),
                axis=-1,
            )
            # print(f"{dlogp=}")
        active_trials = active_model[range(1, self.N_f + 1), ...].sum(axis=-1)
        # assert np.all(dlogp>0), f'dlogp should be positive, {dlogp=}'

        reliability = active_trials / self.n_samples
        # print(f"reliability sigmoid: {reliability_sigmoid(reliability,self.n_samples,threshold_active_trials)}")
        reliability_penalty = (
            reliability_sigmoid(reliability,self.n_samples,threshold_active_trials)
            # np.maximum(threshold_active_trials-active_trials,0)/threshold_active_trials * \
            * dlogp
        ).sum(axis=0)
        # print(f"{reliability_penalty=}")

        self.timeit("reliability penalty")
        return reliability_penalty
    
    

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


def reliability_sigmoid(r,n,n_threshold):
    return 1 - 1 / (1 + np.exp(-n * (r - n_threshold/n)))
        
    

from .HierarchicalBayesModel.NestedSamplingMethods import run_sampling
from .result_structures import PlaceFieldInferenceResults, handover_inference_results

def model_comparison(
    event_counts,
    T,
    mode="dynesty",
    show_status=False,
    limit_execution_time=None,
    nP=1
):
    t_start = time.time()

    vectorized = mode == "ultranest"

    HBI = HierarchicalBayesInference(logLevel="ERROR")
    HBI.prepare_data(
        event_counts,
        T,
        iter_dims=False,
        dimension_names=["trials", "position_bins"],
    )
    # HBI.set_priors(N_f=2)
    res = PlaceFieldInferenceResults(n_bin=HBI.n_bin, n_trials=HBI.n_samples)

    # inference_results = build_results(n_bin=HBI.n_bin, n_trials=HBI.n_samples, modes=["bayesian"])
    inference_results = {
        "field_models": {
            "logz": np.full((3,2),np.nan)
        }
    }

    if limit_execution_time:

        def handler(signum, frame):
            print("Forever is over!")
            raise TimeoutException("end of time")

        signal.signal(signal.SIGALRM, handler)

    if (event_counts > 0).sum() < 10:
        print("Not enough instances of activity detected")
        return None

    previous_logz = -np.inf
    for n_field in range(2 + 1):
        if show_status:
            print(f"\n{n_field=}\n")

        try:
            if limit_execution_time:
                signal.alarm(limit_execution_time)
            
            HBI.set_priors(N_f=n_field)
            res.build_results(HBI.priors)

            my_prior_transform = HBI.set_prior_transform(vectorized=vectorized)
            my_likelihood = HBI.set_logp_func(vectorized=vectorized)

            tmp_results, _ = run_sampling(
                my_prior_transform,
                my_likelihood,
                HBI.parameter_names_all,
                nP=nP,
                n_live=100, periodic=HBI.periodic,
                mode=mode
            )
            res.store_inference_results(tmp_results,HBI.parameter_names_all,HBI.periodic,HBI.set_logp_func(vectorized=False))    # this includes calculation of reliability and active trials

            inference_results["field_models"][n_field] = res.fields
            # handover_inference_results(res.fields, inference_results["bayesian"]["field_models"][n_field])

            # hand over logz to joint results for easier comparison
            inference_results["field_models"]["logz"][n_field,:] = inference_results["field_models"][n_field]["logz"]

            if limit_execution_time:
                signal.alarm(0)

            ## 3 degrees of freedom, as statistic depends on difference of dof between models
            if chi2.sf(-2*(previous_logz - inference_results["field_models"]["logz"][n_field,0]), 3) > 0.01:
                break

            N_f = n_field

            previous_logz = inference_results["field_models"]["logz"][n_field,0]
        except Exception as exc:
            print("Exception:", exc)
            break
        
    HBI.set_priors(N_f=2)
    res.build_results(priors=HBI.priors)
    inference_results["fields"] = res.fields
    inference_results["fields"] = handover_inference_results(inference_results["field_models"][N_f], inference_results["fields"])
    # inference_results["bayesian"]["field_models"][n_field]
    # self.calculate_general_statistics()

        # if plot:
        #     self.display_results()

    try:
        string_out = f"Model comparison finished after {time.time() - t_start:.2f}s with evidences: "
        for f in range(2 + 1):
            if not np.isnan(inference_results["field_models"]["logz"][f, 0]):
                string_out += f"\t {f=} {'*' if f==inference_results['fields']['n_modes'] else ''}, logz={inference_results['field_models']['logz'][f,0]:.2f}"
    except Exception as exc:
        string_out = f"Building result string failed: {exc}"
        # else:
        #     self.calculate_general_statistics(which=["firingstats"])

    print(string_out)

    return inference_results


def norm_cdf(x, mu, sigma):
    return 0.5 * (1.0 + erf((x - mu) / (np.sqrt(2) * sigma)))


@dataclass
class place_field:
    A: float
    sigma: float
    theta: float
    reliability: float = 1.

    def cast_to_array(self):
        for attr in attributes(self):
            setattr(self,attr.name,np.atleast_1d(getattr(self,attr.name)))


class TimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        pass


# import math

# def _phi(z):
#     """Standard normal CDF."""
#     return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# def _normal_cdf(x, mu, sigma):
#     return _phi((x - mu) / sigma)

# def _normal_pdf(x, mu, sigma):
#     return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))

# def gaussian_overlap(mu1, sigma1, mu2, sigma2, *, eps=1e-12):
#     """
#     Overlapping coefficient (OVL) between N(mu1, sigma1^2) and N(mu2, sigma2^2).
#     Returns a value in [0, 1]. No external dependencies.
#     """
#     if sigma1 <= 0 or sigma2 <= 0:
#         raise ValueError("sigma1 and sigma2 must be positive")

#     # Equal-variance shortcut (and numerically near-equal)
#     if abs(sigma1 - sigma2) <= eps * max(sigma1, sigma2):
#         d = abs(mu1 - mu2) / (2.0 * sigma1)
#         return max(0.0, min(1.0, 2.0 * _phi(-d)))

#     # Solve f1(x) = f2(x) for intersection points (quadratic)
#     # ((x-m2)^2)/s2^2 - ((x-m1)^2)/s1^2 = 2 ln(s2/s1)
#     s1, s2 = sigma1, sigma2
#     m1, m2 = mu1, mu2

#     a = 1.0 / (s2 * s2) - 1.0 / (s1 * s1)
#     b = -2.0 * (m2 / (s2 * s2) - m1 / (s1 * s1))
#     c = (m2 * m2) / (s2 * s2) - (m1 * m1) / (s1 * s1) - 2.0 * math.log(s2 / s1)

#     # If 'a' is tiny, the curves are near-equal variance; fall back.
#     if abs(a) < 1e-20:
#         d = abs(mu1 - mu2) / (2.0 * ((s1 + s2) / 2.0))
#         return max(0.0, min(1.0, 2.0 * _phi(-d)))

#     disc = b * b - 4.0 * a * c
#     if disc < 0 and disc > -1e-14:  # clamp tiny negatives due to rounding
#         disc = 0.0
#     if disc < 0:
#         # Numerically weird case: densities might be almost identical; return ~1
#         return 1.0

#     sqrt_disc = math.sqrt(disc)
#     xL = (-b - sqrt_disc) / (2.0 * a)
#     xR = (-b + sqrt_disc) / (2.0 * a)
#     if xL > xR:
#         xL, xR = xR, xL

#     # Decide which density is the minimum on (xL, xR) by checking the midpoint
#     m = 0.5 * (xL + xR)
#     f1m = _normal_pdf(m, m1, s1)
#     f2m = _normal_pdf(m, m2, s2)

#     if f1m <= f2m:
#         # f1 is the smaller in the middle; f2 is smaller outside
#         ovl = (_normal_cdf(xR, m1, s1) - _normal_cdf(xL, m1, s1)) \
#             + (_normal_cdf(xL, m2, s2) + (1.0 - _normal_cdf(xR, m2, s2)))
#     else:
#         # swap roles
#         ovl = (_normal_cdf(xR, m2, s2) - _normal_cdf(xL, m2, s2)) \
#             + (_normal_cdf(xL, m1, s1) + (1.0 - _normal_cdf(xR, m1, s1)))

#     # Clip to [0,1] to tame numerical noise
#     return max(0.0, min(1.0, ovl))