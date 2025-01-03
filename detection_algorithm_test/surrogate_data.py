##% inputs

## p      - probability of being a place cell
## place_cell_distribution  - distribution of PC locations (default: 'uniform')

## activity - structure with all kind of information about the firing activity
##      offset      - height and variance
##      sigma       - width and variance of the firing field
##      amplitude   - activity amplitude and variance of amplitude
##      num_fields
##
## track    - structure with some information about the track
##      length      - length of the track
##      nbin        - number of bins

import numpy as np
import matplotlib.pyplot as plt
import random, time, tqdm
import multiprocessing as mp
from functools import partial

from utils import intensity_model_from_position, lognorm_paras


class SurrogateData:

    def __init__(
        self, nCells, track, place_field_parameter, behavior, place_cell_probability
    ):
        """
        ToDo:
                - how to get rid of track (do I want to?)
                - implement more clean version of generate_activity
                - rename activity -> place field parameters
        """

        self.place_cell_status = np.zeros(nCells).astype("bool")
        self.options = {"place_cell_distribution": "uniform"}

        # self.behavior = behavior
        self.behavior = {
            "active": behavior["active"],
            "position": behavior["binpos_raw"],
            "time": behavior["time_raw"],
            "T": behavior["time_raw"][-1],
            "trial_ct": behavior["trials"]["ct"],
            "trial_start": np.hstack(
                [
                    np.where(behavior["active"])[0][behavior["trials"]["start"][:-1]],
                    len(behavior["time_raw"]),
                ]
            ),
            "nbin": track["nbin"],
        }

        self.nCells = nCells
        self.nbin = track["nbin"]

        # self.intensity_model = intensity_model

        self.activity = np.zeros((nCells, len(self.behavior["time"])))
        self.field_activation = np.zeros(
            (nCells, place_field_parameter["n_fields"], self.behavior["trial_ct"]),
            "bool",
        )

        self.tuning_curve_parameter = []
        for n in range(nCells):

            self.tuning_curve_parameter.append(
                self.set_PC_field(place_field_parameter, place_cell_probability)
            )

            if self.tuning_curve_parameter[n]["n_fields"] > 0:
                self.place_cell_status[n] = True

            # self.rate_poisson_process.append(self.set_TC_fun(self.tuning_curve_parameter[n]))

    def set_PC_field(self, place_field_parameter, place_cell_probability):
        """
        if non-uniform distribution of place cells is introduced, need to implement different algorithm for drawing locations.
        Could be, that each parameter is drawn from a different distribution, as specified in the input place_field_parameter-dictionary
        """
        tuning_curve_parameter = {}
        if (
            self.options["place_cell_distribution"] == "uniform"
        ):  ## uniform distribution

            tuning_curve_parameter["n_fields"] = (
                random.randint(1, place_field_parameter["n_fields"])
                if (random.random() < place_cell_probability)
                else 0
            )

            tuning_curve_parameter["A0"] = draft_para(place_field_parameter["A0"])

            if tuning_curve_parameter["n_fields"]:
                tuning_curve_parameter["PF"] = []

            for field in range(tuning_curve_parameter["n_fields"]):

                tuning_curve_parameter["PF"].append({})
                for key in ["theta", "A", "sigma", "reliability"]:
                    tuning_curve_parameter["PF"][field][key] = draft_para(
                        place_field_parameter[key]
                    )  ## uniform distribution

        elif self.options["place_cell_distribution"] == "salientgauss":
            ## gaussian around salient locations provided
            assert False, "not implemented yet"

        elif self.options["place_cell_distribution"] == "manual":
            ## according to provided distribution
            assert False, "not implemented yet"

        return tuning_curve_parameter

    def generate_activity_all(self, nP=0):
        """
        wrapper for running generate_activity for all neurons, possibly in parallel
        """

        t_start = time.time()
        generate_activity_fun = partial(generate_activity, behavior=self.behavior)
        if nP > 0:

            pool = mp.Pool(nP)
            results = pool.map(generate_activity_fun, self.tuning_curve_parameter)
            # print(results)
            for (S, act), n in zip(results, range(self.nCells)):
                self.activity[n, :] = S
                self.field_activation[
                    n, : self.tuning_curve_parameter[n]["n_fields"], :
                ] = act

            print(">>> all done. time passed: %6.4g secs <<<" % (time.time() - t_start))
        else:
            ### generate activity for neurons

            for n, tcp in tqdm.tqdm(enumerate(self.tuning_curve_parameter)):

                self.activity[n, :], self.field_activation[n, : tcp["n_fields"], :] = (
                    generate_activity_fun(tcp, plt_bool=False)
                )

    # def generate_activity(self,n):

    def plot_activity(self, n):

        intensity_model = lambda x, parameter, fields: intensity_model_from_position(
            x=x, parameter=parameter, n_x=self.behavior["nbin"], fields=fields
        )

        # model_intensity_at_frame = intensity_model(behavior['position'],tuning_curve_parameter,range(self.tuning_curve_parameter[n]['n_fields']))

        dwelltime, _ = np.histogram(
            self.behavior["position"],
            bins=np.linspace(0, self.behavior["nbin"], self.behavior["nbin"] + 1),
        )
        fmap = get_firingmap(
            self.activity[n, :], self.behavior["position"], dwelltime / 15.0
        )

        spike_frames = np.where(self.activity[n, :])[0]
        spikes = self.activity[n, spike_frames]
        spike_times = self.behavior["time"][spike_frames]
        ISI = np.diff(spike_times)

        fig = plt.figure(figsize=(10, 6))

        ax_position = fig.add_subplot(311)
        ax_APs = fig.add_subplot(312, sharex=ax_position)
        ax_fmap = fig.add_subplot(325)
        ax_ISI = fig.add_subplot(326)

        # ax_intensity = plt.subplot(323)
        for trial in range(self.behavior["trial_ct"]):
            for f in range(self.tuning_curve_parameter[n]["n_fields"]):
                if self.field_activation[n, f, trial]:
                    # print('t:',behavior['time'][[trial_starts[trial],trial_starts[trial+1]]])
                    # for axx in [ax_position,ax_intensity]:
                    ax_position.axvspan(
                        self.behavior["time"][self.behavior["trial_start"][trial]],
                        self.behavior["time"][
                            self.behavior["trial_start"][trial + 1] - 1
                        ],
                        color="tab:green",
                        alpha=0.3,
                    )

        ax_position.plot(self.behavior["time"], self.behavior["position"], "k")
        ax_position.scatter(
            spike_times,
            self.behavior["position"][spike_frames],
            color="r",
            marker="o",
            s=2 + 2 * spikes,
        )

        ax_APs.plot(self.behavior["time"], self.activity[n, :], "k")
        # ax_intensity.plot(self.behavior['time'],model_intensity_at_frame,'k-')
        # ax_intensity.scatter(T_AP,model_intensity_at_frame[AP_frame],color='r',marker='o',s=1)

        x = np.linspace(0, self.behavior["nbin"], 101)
        ax_fmap.plot(
            x,
            intensity_model(
                x,
                self.tuning_curve_parameter[n],
                range(self.tuning_curve_parameter[n]["n_fields"]),
            ),
            "k",
        )
        ax_fmap.plot(fmap)

        ax_ISI.hist(ISI, np.linspace(0, 2, 101))

        for axx in [ax_position, ax_APs, ax_fmap, ax_ISI]:
            axx.spines[["top", "right"]].set_visible(False)

        plt.setp(ax_position, xlim=[0, self.behavior["T"]], ylabel="position")
        plt.setp(ax_APs, ylabel="# spikes", xlabel="time [s]")

        plt.setp(ax_fmap, xlabel="position", ylabel="firing rate")
        plt.setp(ax_ISI, xlabel="ISI [s]", ylabel="count")

        plt.suptitle(f"neuron {n}")
        plt.tight_layout()
        plt.show(block=False)
        # plt.close('all')


def generate_activity(
    tuning_curve_parameter, behavior, AP_mode="constant", plt_bool=False
):
    """
    generating a nonhomogeneous poisson process (pp)
    on [0,T] with intensity function model_intensity
    """

    intensity_model = lambda x, parameter, fields: intensity_model_from_position(
        x=x, parameter=parameter, n_x=behavior["nbin"], fields=fields
    )

    # model_intensity = lambda x: intensity_model(x,tuning_curve_parameter)

    ## create homogeneous poisson process at maximum intensity, spanning the whole experimental time T

    ## define rate as maximum rate of homogeneous pp

    if tuning_curve_parameter["n_fields"] > 0:
        max_intensity = 0
        for field in tuning_curve_parameter["PF"]:
            max_intensity = max(
                max_intensity, tuning_curve_parameter["A0"] + field["A"]
            )
    else:
        max_intensity = tuning_curve_parameter["A0"]

    u = np.random.rand(int(np.ceil(1.1 * behavior["T"] * max_intensity)))
    t_AP = np.cumsum(-(1 / max_intensity) * np.log(u))
    t_AP = t_AP[t_AP < behavior["T"]]  # cut off at T_end

    ## identify index of discretized time array, at which spike occurs
    nAP = len(t_AP)  # number of spikes
    AP_frame = np.zeros(nAP, "int")
    for AP, i in zip(t_AP, range(nAP)):
        AP_frame[i] = np.argmin(abs(AP - behavior["time"]))
    T_AP = behavior["time"][AP_frame]  # timepoints of homogeneous pp (discrete time)

    if tuning_curve_parameter["n_fields"] > 0:

        idx_keep = np.zeros(nAP, "bool")

        field_activation = np.zeros(
            (tuning_curve_parameter["n_fields"], behavior["trial_ct"]), "bool"
        )

        for trial in range(behavior["trial_ct"]):

            # get some trial information for better readability
            trial_start = behavior["trial_start"][trial]
            trial_end = behavior["trial_start"][trial + 1]
            # print(f'{trial=} trial_start: {trial_start}, trial_end: {trial_end}')

            AP_idx_this_trial = np.where(
                (AP_frame >= trial_start) & (AP_frame < trial_end)
            )[0]
            # print(f'{AP_idx_this_trial=}')
            # print(f'{AP_frame[AP_idx_this_trial]=}')
            if len(AP_idx_this_trial) == 0:
                continue
            first_AP_idx = AP_idx_this_trial[0]
            last_AP_idx = AP_idx_this_trial[-1]
            N_APs = len(AP_idx_this_trial)

            # print(f'{first_AP_idx=}, {last_AP_idx=}, {N_APs=}')

            position_this_trial = behavior["position"][trial_start:trial_end]
            # model_intensity_at_frame_this_trial = model_intensity_at_frame[trial_start:trial_end]

            ## decide, which fields are active in this trial
            active_fields = []
            for f, field in enumerate(tuning_curve_parameter["PF"]):
                if random.random() < field["reliability"]:
                    active_fields.append(f)
                    field_activation[f, trial] = True
                    # ct_activate[f] += 1

            model_intensity_at_frame_this_trial = intensity_model(
                position_this_trial, tuning_curve_parameter, active_fields
            )

            model_intensity_at_spikes_this_trial = model_intensity_at_frame_this_trial[
                AP_frame[AP_idx_this_trial] - AP_frame[first_AP_idx]
            ]  # evaluates intensity function at homogeneous pp points

            idx_keep_trial = np.random.rand(N_APs) < (
                model_intensity_at_spikes_this_trial / max_intensity
            )

            idx_keep[first_AP_idx : last_AP_idx + 1] = idx_keep_trial

        AP_frame = AP_frame[idx_keep]

        t_AP = t_AP[
            idx_keep
        ]  ## filter out some points with prob according to ratio of nonhomo/homo pp
        T_AP = T_AP[idx_keep]

    nAP = len(t_AP)

    ## finally, adding up spike contributions at each timepoint (as some inter-spike intervals could be <dt)
    activity = np.zeros_like(behavior["time"])

    if AP_mode == "constant":
        width = 0.5
        AP_heights = 1 - width + 2 * width * np.random.random(nAP)
    elif AP_mode == "lognorm":
        mu_lognorm, sigma_lognorm = lognorm_paras(1, 1)
        AP_heights = np.random.lognormal(mu_lognorm, sigma_lognorm, nAP)
    for AP, height in zip(AP_frame.astype("int"), AP_heights):
        activity[AP] += height

    return activity, field_activation


def draft_para(in_range, n=1):
    return (
        in_range[0] + (in_range[1] - in_range[0]) * random.random()
    )  # np.random.rand(n)


def get_firingmap(S, binpos, dwelltime=None, nbin=None):

    if not nbin:
        nbin = np.max(binpos) + 1

    ### calculates the firing map
    spike_times = np.where(S)
    spikes = S[spike_times]
    binpos = binpos[spike_times]  # .astype('int')

    firingmap = np.zeros(nbin)
    for p, s in zip(binpos, spikes):  # range(len(binpos)):
        firingmap[p] = firingmap[p] + s

    if not (dwelltime is None):
        firingmap = firingmap / dwelltime
        firingmap[dwelltime == 0] = np.NaN

    return firingmap
