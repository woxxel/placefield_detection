import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.ndimage import binary_opening, gaussian_filter1d as gauss_filter

from .utils_various import gauss_smooth
from .utils_io import load_data

def prepare_behavior(
    position: np.ndarray,
    time: np.ndarray,
    only_active: bool = True,
    nbin: int = 100,
    environment_length = 120,
    f: float = 15.0,
    T = None,
    calculate_performance: bool = False,
    rw_loc_in = None,
    **kwargs
):
    """
    prepares behavior given by time and position data
    Requires file to contain a dictionary with values for each frame, aligned to imaging data:
        * position  - mouse position
        * time      - time in seconds

    calculates:
        * active    - boolean array defining active frames (included in analysis)
    """

    data = {
        "nbin": nbin,
        "environment_length": float,
        "time": np.ndarray,
        "position": np.ndarray,
        "velocity": np.ndarray,
        "active": np.ndarray,
        "trials": {},
        "nFrames": int,
        "dwelltime": np.ndarray,
    }

    if T is None:
        T = time.shape[0]

    binpos, length_tmp = get_binpos(position, nbin)
    data["environment_length"] = (
        length_tmp if environment_length is None else environment_length
    )

    velocity = get_velocity(position, data["environment_length"])
    data["active"] = get_active(velocity)

    data["time_original"] = time.copy()
    data["position_original"] = binpos.copy()
    data["velocity_original"] = velocity.copy()

    if only_active:
        data["time"] = time[data["active"]]
        data["position"] = binpos[data["active"]]
        data["velocity"] = velocity[data["active"]]
    else:
        data["time"] = time.copy()
        data["position"] = binpos.copy()
        data["velocity"] = velocity.copy()

    if len(data["position"]) > 0:
        data["trials"] = get_trial_data(data["position"], nbin, f, **kwargs)

        if only_active:
            ### preparing data for active periods, only
            correct_active_from_trials(data["active"], data["trials"])

            data["trials"]["start"] -= data["trials"]["start"][0]

            data["time"] = time[data["active"]]
            data["position"] = binpos[data["active"]]
            data["velocity"] = velocity[data["active"]]

        data["dwelltime"] = get_dwelltime(data["position"], nbin, f)

        if calculate_performance and rw_loc_in is not None:
            rw_pos = rw_loc_in * nbin
            # try:
            data["performance"] = get_performance(
                binpos, velocity, time, rw_pos, 0, nbin, f, **kwargs
            )
            # except:
            #     pass

    data["nFrames"] = len(data["position"])

    return data


def prepare_behavior_from_file(
    path,
    only_active=True,
    nbin=100,
    environment_length=120,
    f=15.0,
    T=None,
    calculate_performance=False,
    **kwargs
):
    """
    loads behavior from specified path
    Requires file to contain a dictionary with values for each frame, aligned to imaging data:
        * time      - time in seconds
        * position  - mouse position
        * active    - boolean array defining active frames (included in analysis)

    """
    data = load_data(path)
    # with open(path, "rb") as f_open:
    #     data = pickle.load(f_open)
    return prepare_behavior(
        data["position"],
        data["time"],
        only_active,
        nbin,
        environment_length,
        f,
        T,
        calculate_performance,
        data["reward_location"],
        **kwargs
    )


def get_binpos(position: np.ndarray, nbin: int) -> tuple[np.ndarray, float]:

    ## get range of values
    min_val, max_val = np.nanpercentile(position, (0.1, 99.9))
    environment_length = max_val - min_val

    position[np.isnan(position)] = min_val

    binpos = np.minimum(
        (position - min_val) / environment_length * nbin, nbin - 1
    ).astype("int")
    binpos = signal.medfilt(binpos, kernel_size=3)

    return binpos, environment_length


def get_dwelltime(position, nbin, f):

    bin_array = np.arange(nbin + 1) - 0.5

    ## define dwelltimes
    dwelltime = np.histogram(position, bin_array)[0] / f

    return dwelltime


def get_velocity(position, environment_length, f=15.0, speed_gauss_sd=4):
    """
    calculates velocity from position data
    """
    speed_gauss_sd = f / 5.0
    raw_velocity = gauss_filter(
        np.maximum(0, np.diff(position, prepend=position[0])), speed_gauss_sd
    ).astype("float64")

    np.clip(raw_velocity, 0, np.percentile(raw_velocity,99.), out=raw_velocity)

    min_val, max_val = np.nanpercentile(position, (0.1, 99.9))
    absolute_distance = max_val - min_val

    return raw_velocity * f * environment_length / absolute_distance


def get_active(velocity, speed_thr=2.0, binary_morph_width=5):
    """
    returns a boolean array defining active frames (included in analysis)
    """

    inactive = binary_opening(velocity <= speed_thr, np.ones(binary_morph_width)).astype("bool")
    return ~inactive


def get_trial_data(position, nbin, f, partial_threshold=0.6,**kwargs):

    trials = {}
    ## define start points
    trials["start"] = np.hstack(
        [0, np.where(np.diff(position) < (-nbin / 2))[0] + 1, position.shape[0]]
    )
    if not (position[0] < nbin * (1 - partial_threshold)):
        # print("remove partial first trial @", trials["start"][0])
        trials["start"] = trials["start"][1:]

    if not (position[-1] >= nbin * partial_threshold):
        # print("remove partial last trial @", trials["start"][-1])
        trials["start"] = trials["start"][:-1]

    # trials['start_t'] = data['time'][trials['start'][:-1]]
    trials["ct"] = len(trials["start"]) - 1

    ## getting trial-specific behavior data
    trials["dwelltime"] = np.zeros((trials["ct"], nbin))
    trials["nFrames"] = np.zeros(trials["ct"], "int")
    trials["position"] = {}

    for t in range(trials["ct"]):
        trials["position"][t] = position[trials["start"][t] : trials["start"][t + 1]]
        trials["dwelltime"][t, :] = get_dwelltime(trials["position"][t], nbin, f)
        trials["nFrames"][t] = len(trials["position"][t])
    return trials


def correct_active_from_trials(active, trials):
    """
    refines active frames to only include wanted trials
    """

    if trials["start"][-1] == active.sum():
        active_start = np.where(active)[0][trials["start"][0]]
        active_end = np.where(active)[0][-1] + 1
    else:
        trials_start = np.where(active)[0][trials["start"]]
        active_start = trials_start[0]
        active_end = trials_start[-1]

    active[:active_start] = False
    active[active_end:] = False

    ### necessary?
    # return active, trials


def get_performance(
    position, velocity, time, rw_pos, rw_delay, nbin, f, plt_trials=False, plt_bool=False
):

    trials = get_trial_data(position, nbin, f)

    ## get behavior performance
    rw_end = rw_pos + nbin / 5
    rw_delay = 0

    range_approach = [-2, 4]  ## in secs
    ra = range_approach[1] - range_approach[0]

    ra_frames = int(f * ra)
    ra_idxs = np.arange(int(range_approach[0] * f), int(range_approach[1] * f))
    ra_arr = np.linspace(range_approach[0], range_approach[1], ra_frames)

    vel_max = np.ceil(np.max(velocity) / 10) * 10
    vel_arr = np.linspace(vel_max / 50, vel_max, 51)

    hist = gauss_smooth(np.histogram(velocity, vel_arr)[0], 2, mode="nearest")

    try:
        vel_run_idx = signal.find_peaks(hist, distance=10, prominence=vel_max * 0.3)[0][-1]
        vel_min_idx = signal.find_peaks(-hist, distance=5)[0]
        vel_min_idx = vel_min_idx[vel_min_idx < vel_run_idx][-1]
        vel_thr = vel_arr[vel_min_idx]

    except:
        vel_thr = np.median(velocity[velocity > 0])

    performance = {}
    performance["RW_reception"] = np.zeros(trials["ct"], "bool")
    performance["RW_frame"] = np.zeros(trials["ct"], "int")

    performance["slowDown"] = np.zeros(trials["ct"], "bool")
    performance["frame_slowDown"] = np.zeros(trials["ct"], "int")
    performance["pos_slowDown"] = np.full(trials["ct"], np.nan)
    performance["t_slowDown_beforeRW"] = np.full(trials["ct"], np.nan)

    performance["RW_approach_time"] = np.zeros((trials["ct"], int(ra * f)))
    performance["RW_approach_space"] = np.full((trials["ct"], nbin), np.nan)

    if plt_trials:
        ncols = min(trials["ct"], 5)
        fig_trial_time, axes_trial_time = plt.subplots((trials["ct"]-1) // ncols + 1, ncols)
        fig_trial_position, axes_trial_position = plt.subplots((trials["ct"]-1) // ncols + 1, ncols)

        if (trials["ct"]-1) // ncols == 0:
            axes_trial_time = axes_trial_time[np.newaxis,:]
            axes_trial_position = axes_trial_position[np.newaxis,:]

    for t in range(trials["ct"]):
        pos_trial = position[trials["start"][t] : trials["start"][t + 1]].astype("int")
        vel_trial = velocity[trials["start"][t] : trials["start"][t + 1]]
        time_trial = time[trials["start"][t] : trials["start"][t + 1]].copy()
        time_trial -= time_trial[0]  # start at 0
        for j, p in enumerate(range(nbin)):
            performance["RW_approach_space"][t, j] = vel_trial[pos_trial == p].mean()

        idx_enterRW = np.where(pos_trial > rw_pos)[0][
            0
        ]  ## find, where first frame within rw position is
        idx_RW_reception = int(idx_enterRW + rw_delay * f)

        idx_exitRW = np.where(pos_trial > rw_end)[0]  ## find, where first frame after rw position is

        if len(idx_exitRW) > 0:
            idx_exitRW = idx_exitRW[0]
        else:
            idx_exitRW = pos_trial.shape[0] - 1

        # print(f"Trial {t}: RW reception at {idx_RW_reception}, exit at {idx_exitRW}")
        # mean_vel = np.mean(vel_trial[idx_RW_reception:idx_exitRW])
        # print(f"Trial {t}: Mean velocity from RW reception to exit: {mean_vel:.2f} cm/s")

        performance["RW_approach_time"][t, :] = vel_trial[ra_idxs + idx_RW_reception]

        if plt_trials:
            ax_pos = axes_trial_position[t // ncols, t % ncols]
            ax_time = axes_trial_time[t // ncols, t % ncols]

            ax_pos.plot(performance["RW_approach_space"][t, :],"k-",linewidth=0.5)
            ax_time.plot(time_trial,vel_trial, "k-",linewidth=0.5)

            ax_pos.axhline(vel_thr, color="k", linestyle="--")
            ax_pos.plot(rw_pos, vel_trial[idx_enterRW], "go",markersize=5)
            ax_pos.plot(
                rw_end, performance["RW_approach_space"][t, int(rw_end)], "go",markersize=5
            )

            ax_time.axhline(vel_thr, color="k", linestyle="--",linewidth=0.5)
            ax_time.plot(time_trial[idx_enterRW], vel_trial[idx_enterRW], "go",markersize=3,label="reward area")
            ax_time.plot(time_trial[idx_exitRW], vel_trial[idx_exitRW], "go",markersize=3)

        time_slowed_down = (vel_trial[idx_enterRW:idx_exitRW] < vel_thr).sum()/f
        # print(f"Trial {t}: Time slowed down: {time_slowed_down:.2f} s")
        if (pos_trial[idx_RW_reception] < rw_end) and (time_slowed_down > 0.5):
            performance["RW_frame"][t] = trials["start"][t] + idx_RW_reception
            performance["RW_reception"][t] = True

            ax_time.plot(time_trial[idx_enterRW:idx_exitRW],vel_trial[idx_enterRW:idx_exitRW], "g-",linewidth=1)

            # print(performance["RW_approach_space"][t,:].shape)
            ax_pos.plot(range(int(rw_pos),int(rw_end)),performance["RW_approach_space"][t,int(rw_pos):int(rw_end)], "g-",linewidth=1)

            idx_trough_tmp = signal.find_peaks(
                -vel_trial, prominence=2, height=-vel_thr, distance=f
            )[0]

            # print('trough_tmp: ',idx_trough_tmp)
            idx_trough_tmp = idx_trough_tmp[idx_trough_tmp > idx_enterRW]

            # if plt_trials:
            #     for i in idx_trough_tmp:
            #         ax_pos.plot(pos_trial[i], vel_trial[i], "go")

            if len(idx_trough_tmp) > 0:
                # idx_trough = idx_enterRW + idx_trough_tmp[0]
                idx_trough = idx_trough_tmp[0]
                ### slowing down should occur before this - defined by drop below threshold velocity
                slow_down = np.where(
                    (vel_trial[:idx_trough] > vel_thr)
                    & (pos_trial[:idx_trough] <= rw_end)
                    & (pos_trial[:idx_trough] > 5)
                )[0][-1]

                if plt_trials:
                    ax_pos.plot(
                        pos_trial[slow_down], vel_thr, "rx", markersize=5, label="slow down"
                    )
                    # ax_pos.plot(
                    #     pos_trial[idx_trough], vel_thr, "ro", markersize=3
                    # )

                    ax_time.plot(
                        time_trial[slow_down], vel_trial[slow_down], "rx", markersize=5,label="slow down"
                    )
                    # ax_time.plot(
                    #     time_trial[idx_trough], vel_trial[idx_trough], "ro", markersize=3
                    # )

                # print('velocity @ trial ',t,':  ',vel_trial[slow_down+1:int(slow_down+1+f)].mean(),vel_thr)
                if (
                    vel_trial[slow_down + 1 : int(slow_down + 1 + f)].mean() < vel_thr
                ):  # vel_trial[slow_down+1]<vel_thr:
                    performance["slowDown"][t] = True
                    performance["frame_slowDown"][t] = trials["start"][t] + slow_down
                    performance["pos_slowDown"][t] = pos_trial[slow_down]
                    performance["t_slowDown_beforeRW"][t] = (
                        time_trial[idx_RW_reception] - time_trial[slow_down]
                    )
    if plt_trials:
        for axx in axes_trial_time.flatten():
            axx.spines[["top", "right"]].set_visible(False)
        for axx in axes_trial_time[:,1:].flatten():
            axx.set_yticklabels([])

        # Create a big invisible axis for shared labels
        big_ax = fig_trial_time.add_subplot(111, frameon=False)
        big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        big_ax.grid(False)
        plt.setp(big_ax, xlabel="Time (s)", ylabel="Velocity (cm/s)")
        big_ax.spines[["top","right","bottom","left"]].set_visible(False)

        fig_trial_time.tight_layout()

        print(nbin)
        for axx in axes_trial_position.flatten():
            plt.setp(axx,xticks=range(0,nbin,10),xlim=[0,nbin])
            axx.spines[["top", "right"]].set_visible(False)
        for axx in axes_trial_position[:-1,:].flatten():
            axx.set_xticklabels([])
        for axx in axes_trial_position[:,1:].flatten():
            axx.set_yticklabels([])

        axes_trial_time[0][0].legend(fontsize=8)
        # Create a big invisible axis for shared labels
        big_ax = fig_trial_position.add_subplot(111, frameon=False)
        big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        big_ax.grid(False)
        plt.setp(big_ax, xlabel="Position (bin)", ylabel="Velocity (cm/s)")
        big_ax.spines[["top","right","bottom","left"]].set_visible(False)

        fig_trial_position.tight_layout()

        plt.show(block=False)

    if plt_bool:

        # fig,ax = plt.subplots(2,1,figsize=(8,5))
        # ax[0].plot(time,position)
        # ax[1].plot(time,velocity)
        # # for s in trial_start:
        # #     ax[0].axvline(time[s],color='r')
        # #     ax[1].axvline(time[s],color='r')
        # plt.show(block=False)

        # fig,ax = plt.subplots(2,1,figsize=(8,5))
        # # print(data['bi'])
        # ax[0].plot(data['position'])
        # ax[1].plot(data['velocity'])
        # for s in data['trials']['start']:
        #     ax[0].axvline(s,color='r')
        #     ax[1].axvline(s,color='r')
        # plt.show(block=False)

        fig,axes = plt.subplots(2,2, figsize=(8, 5))

        # ax = fig.add_subplot(221)
        axes[0][0].plot(
            ra_arr, performance["RW_approach_time"].T, color=[0.5, 0.5, 0.5], alpha=0.5
        )
        axes[0][0].plot(ra_arr, performance["RW_approach_time"].mean(0), color="k")
        axes[0][0].plot(
            -performance["t_slowDown_beforeRW"][performance["slowDown"]],
            velocity[performance["frame_slowDown"][performance["slowDown"][:]]],
            "rx",
        )
        axes[0][0].axhline(vel_thr, color="k", linestyle="--", linewidth=0.5)
        plt.setp(axes[0][0],xlabel="Time from RW entry (s)", ylabel="Velocity (cm/s)",xlim=range_approach)
        # plt.xlim(range_approach)

        axes[0][1].plot(
            np.linspace(0, nbin - 1, nbin),
            performance["RW_approach_space"].T,
            color=[0.5, 0.5, 0.5],
            alpha=0.5,
        )
        axes[0][1].plot(
            np.linspace(0, nbin - 1, nbin),
            np.nanmean(performance["RW_approach_space"], 0),
            color="k",
        )
        axes[0][1].plot(
            performance["pos_slowDown"][performance["slowDown"]],
            velocity[performance["frame_slowDown"][performance["slowDown"][:]]],
            "rx",
        )
        axes[0][1].plot([0, nbin], [vel_thr, vel_thr], "k--", linewidth=0.5)
        plt.setp(axes[0][1], xlabel="Position (bin)", ylabel="Velocity (cm/s)")

        axes[1][0].hist(velocity, np.linspace(vel_max / 10, vel_max, 51))
        axes[1][0].plot(np.linspace(vel_max / 10, vel_max, 50), hist)
        axes[1][0].axvline(vel_thr, color="k",linestyle="--",label="slow down threshold")
        plt.setp(axes[1][0], xlabel="Velocity (cm/s)", ylabel="Count")

        for axx in axes.flatten():
            axx.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.show(block=False)

    return performance


def plot_behavior(behavior,only_active=False):

    fig,axes = plt.subplots(2,1,figsize=(6,3))
    # plt.plot(behavior["position"], label="Position")
    # plt.plot(behavior["velocity"], label="Velocity")
    axes[0].plot(
        behavior["time_original"],
        behavior["position_original"],
        "k.",
        linestyle="",
        markersize=2,
        markeredgecolor='none',
        label="Position",
    )
    # if not only_active:
    axes[0].plot(
        behavior["time_original"][~behavior["active"]],
        behavior["position_original"][~behavior["active"]],
        "r.",
        markersize=2,
        markeredgecolor='none',
        label="Inactive",
    )

    axes[0].legend()
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Position (bin)")

    for trial in range(behavior["trials"]["ct"]):
        axes[0].axvline(
            behavior["time"][behavior["trials"]["start"][trial]],
            color="k",
            linestyle="--",
            linewidth=0.5,
        )
    
    axes[1].plot(
        behavior["time_original"],
        behavior["velocity_original"],
        "k-",
        linewidth=0.5,
        # markersize=3,
        # markeredgecolor='none',
        label="Velocity",
    )

    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].spines[["top", "right"]].set_visible(False)
    
    plt.setp(axes[1],xlabel="Time (s)",ylabel="Velocity (cm/s)",ylim=[0,axes[1].get_ylim()[1]])
    plt.show(block=False)
