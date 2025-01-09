import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# sys.path.append(str(Path(__file__).resolve().parents[1]))
# print(str(Path(__file__).resolve().parent))

# from scipy.ndimage import gaussian_filter as gauss_filter
from .utils import gauss_smooth as gauss_filter, model_of_tuning_curve


def build_inference_results_structure(
    n_cells=1,
    N_f=None,
    nbin=None,
    n_trials=None,
    n_steps=100,
    hierarchical=[],
    posterior_arrays=None,
):

    results = {}

    results["status"] = {
        "is_place_cell": {},
        "SNR": np.full(n_cells, np.NaN),
        "r_value": np.full(n_cells, np.NaN),
        # "MI_value": np.full(n_cells, np.NaN),
        ## p-value? z-score? Isec, MI, uMI, etc?
    }

    for method in ["peak", "information", "stability", "bayesian"]:
        results["status"]["is_place_cell"][f"{method}_method"] = np.zeros(
            n_cells, dtype=bool
        )

    results["fields"] = {
        "n_modes": np.zeros(n_cells, dtype=int),
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
        "logz": np.full((n_cells, N_f + 1, 2), np.NaN),
        "active_trials": np.zeros((n_cells, N_f, n_trials)),
        ## need to be calculated extra
        "reliability": np.full((n_cells, N_f), np.NaN),
    }

    ## define ranges for each parameter
    if posterior_arrays is None:
        results["fields"]["x"]["A0"] = np.linspace(0, 10, n_steps + 1)
        results["fields"]["x"]["A"] = np.linspace(0, 100, n_steps + 1)
        results["fields"]["x"]["sigma"] = np.linspace(0, nbin / 2.0, n_steps + 1)
        results["fields"]["x"]["theta"] = np.linspace(0, nbin, n_steps + 1)
    else:
        results["fields"]["x"] = posterior_arrays

    # results["fields"]["x"]["theta"] = np.linspace(0, nbin, nbin + 1)

    key = "A0"
    results["fields"]["parameter"]["global"][key] = np.zeros((n_cells, 3))
    results["fields"]["p_x"]["global"][key] = np.zeros((n_cells, n_steps))

    results["fields"]["parameter"]["local"][key] = (
        np.zeros((n_cells, n_trials, 3)) if key in hierarchical else None
    )
    results["fields"]["p_x"]["local"][key] = (
        np.zeros((n_cells, n_trials, n_steps)) if key in hierarchical else None
    )

    for key in ["theta", "A", "sigma"]:
        n = len(results["fields"]["x"][key]) - 1

        results["fields"]["parameter"]["global"][key] = np.zeros((n_cells, N_f, 3))
        results["fields"]["p_x"]["global"][key] = np.zeros((n_cells, N_f, n))

        results["fields"]["parameter"]["local"][key] = (
            np.zeros((n_cells, N_f, n_trials, 3)) if key in hierarchical else None
        )
        results["fields"]["p_x"]["local"][key] = (
            np.zeros((n_cells, N_f, n_trials, n)) if key in hierarchical else None
        )

    results["firingstats"] = {
        "rate": np.full(n_cells, np.NaN),
        "map": np.full((n_cells, nbin), np.NaN),
        # "CI": np.full((n_cells, 2, nbin), np.NaN),
        "trial_map": np.full((n_cells, n_trials, nbin), np.NaN),
    }

    ## if method is called for nCells = 1, collapse data from first dimension
    def squeeze_deep_dict(d, ax=None):
        for key in d.keys():
            if isinstance(d[key], dict):
                d[key] = squeeze_deep_dict(d[key], ax)
            else:
                if isinstance(d[key], np.ndarray) and (d[key].shape[0] == 1):
                    d[key] = np.squeeze(d[key], axis=ax)
        return d

    if n_cells == 1:
        results = squeeze_deep_dict(results, ax=0)

    return results


def handover_inference_results(
    results_source, results_target, idx, excluded_keys=["x"]
):

    for key in results_source.keys():
        if key in excluded_keys:
            continue

        if isinstance(results_source[key], dict):
            results_target[key] = handover_inference_results(
                results_source[key], results_target[key], idx
            )
        else:
            # print(key, results_source[key], results_target[key])
            if not (results_source[key] is None) and not (results_target[key] is None):
                results_target[key][idx, ...] = results_source[key]

    return results_target


def extract_inference_results(
    results_source, idx, results_target=None, excluded_keys=["x"]
):
    if results_target is None:
        nbin = 40
        _, N_f, n_trials, _ = results_source["fields"]["p_x"]["local"]["theta"].shape
        results_target = build_inference_results_structure(
            n_cells=1, N_f=N_f, nbin=nbin, n_trials=n_trials, hierarchical=["theta"]
        )
        # print(results_target["fields"]["parameter"])

    for key in results_source.keys():
        if key in excluded_keys:
            continue

        if isinstance(results_source[key], dict):
            results_target[key] = extract_inference_results(
                results_source[key], idx, results_target[key]
            )
        else:
            if not (results_source[key] is None) and not (results_target[key] is None):
                results_target[key] = results_source[key][idx, ...]

    return results_target


def display_results(
    results,
    idx=None,
    groundtruth_fields=None,
    groundtruth_activation=None,
):

    if not (idx is None):
        results = extract_inference_results(results, idx)
    # print(results)
    n_trials, nsteps = results["fields"]["p_x"]["local"]["theta"].shape[-2:]
    # nbin = 40
    nbin = 40
    # print(n_trials, nbin)

    if groundtruth_fields:
        groundtruth_N_f = len(groundtruth_fields["PF"])
        field_match = np.full(groundtruth_N_f, -1, "int")
        for f in range(results["fields"]["n_modes"]):
            theta = results["fields"]["parameter"]["global"]["theta"][f, 0]
            # mean[0, self.priors[f"PF{f+1}_theta__mean"]["idx"]]

            for f_truth, field in enumerate(groundtruth_fields["PF"]):

                dTheta = abs(
                    np.mod(theta - field["theta"] + nbin / 2.0, nbin) - nbin / 2.0
                )
                if dTheta <= 5.0:
                    # print("match!", dTheta, theta, field["theta"])
                    field_match[f_truth] = f

    if not (groundtruth_activation is None):
        trial_activation__true_positive = np.zeros((groundtruth_N_f, n_trials), "bool")
        trial_activation__true_negative = np.zeros((groundtruth_N_f, n_trials), "bool")
        trial_activation__false_positive = np.zeros((groundtruth_N_f, n_trials), "bool")
        trial_activation__false_negative = np.zeros((groundtruth_N_f, n_trials), "bool")
        for f in range(groundtruth_N_f):
            if field_match[f] >= 0:
                active_trials = results["fields"]["active_trials"][f, :] > 0.5
                print(f"{active_trials=}")
                print(f"groundtruth={groundtruth_activation[field_match[f], :]}")

                trial_activation__true_positive[f, :] = (
                    groundtruth_activation[field_match[f], :] & active_trials
                )
                trial_activation__true_negative[f, :] = (
                    ~groundtruth_activation[field_match[f], :] & ~active_trials
                )
                trial_activation__false_positive[f, :] = (
                    ~groundtruth_activation[field_match[f], :] & active_trials
                )
                trial_activation__false_negative[f, :] = (
                    groundtruth_activation[field_match[f], :] & ~active_trials
                )
        print(f"{trial_activation__false_negative=}")
        sensitivity = trial_activation__true_positive.sum(axis=1) / (
            trial_activation__true_positive.sum(axis=1)
            + trial_activation__false_negative.sum(axis=1)
        )
        specificity = trial_activation__true_negative.sum(axis=1) / (
            trial_activation__true_negative.sum(axis=1)
            + trial_activation__false_positive.sum(axis=1)
        )

        print(f"{sensitivity=}, {specificity=}")

    if not (groundtruth_fields is None):
        print(
            f"A0: {groundtruth_fields['A0']} vs {results['fields']['parameter']['global']['A0'][0]}"
        )
        for f in range(results["fields"]["n_modes"]):
            if field_match[f] >= 0:
                for key in ["A", "sigma"]:
                    print(
                        f"PF{f+1}_{key}: {groundtruth_fields['PF'][field_match[f]][key]} vs {results['fields']['parameter']['global'][key][f,0]}"
                    )

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, right=0.45)
    # gs_4cols = fig.add_gridspec(6, 4)

    ax_activation = fig.add_subplot(gs[1, :])
    plt.setp(ax_activation, xlabel="trial #", ylabel="$\Theta$")

    ax_theta = fig.add_subplot(gs[2, 0])
    plt.setp(ax_theta, xlabel="$\Theta$", ylabel="p($\Theta$)")

    ax_theta_inactive = fig.add_subplot(gs[2, 1])
    plt.setp(ax_theta_inactive, xlabel="$\Theta$", ylabel="p($\Theta$)")

    comp = not (groundtruth_fields is None)

    if comp:
        for field in groundtruth_fields["PF"]:
            ax_activation.axhline(field["theta"], linestyle="--", color="tab:green")

    col = ["r", "g"]
    if results["fields"]["n_modes"] > 0:
        for f in range(results["fields"]["n_modes"]):
            active_trials = results["fields"]["active_trials"][f, :] > 0.5
            theta_local = results["fields"]["parameter"]["local"]["theta"][f, ...]

            ax_activation.axhline(
                results["fields"]["parameter"]["global"]["theta"][f, 0],
                color="k",
                linestyle="--",
            )

            ax_activation.plot(
                theta_local[:, 0],
                color="r",
                linestyle="-",
                linewidth=0.5,
            )
            # ax_activation.errorbar(
            #     np.where(active_trials)[0],
            #     theta_local[active_trials, 0],
            #     np.abs(
            #         theta_local[active_trials, 0] - theta_local[active_trials, 1:].T
            #     ),
            #     color="k",
            # )  # ,marker='o')

            if comp:
                idx_true_positives = np.where(trial_activation__true_positive[f, ...])[
                    0
                ]
                # print(idx_true_positives)
                # print(theta_vals['mean'][f,idx_true_positives])
                ax_activation.scatter(
                    idx_true_positives,
                    theta_local[idx_true_positives, 0],
                    marker="o",
                    c="tab:green",
                    s=60,
                    alpha=0.6,
                    label="true positives" if f == 0 else None,
                )

                idx_false_negatives = np.where(
                    trial_activation__false_negative[f, ...]
                )[0]
                ax_activation.scatter(
                    idx_false_negatives,
                    np.full(
                        len(idx_false_negatives),
                        results["fields"]["parameter"]["global"]["theta"][f, 0],
                    ),
                    marker="o",
                    c="tab:orange",
                    s=60,
                    alpha=0.6,
                    label="false negatives" if f == 0 else None,
                )

                idx_false_positives = np.where(
                    trial_activation__false_positive[f, ...]
                )[0]
                ax_activation.scatter(
                    idx_false_positives,
                    theta_local[idx_false_positives, 0],
                    marker="o",
                    c="tab:red",
                    s=60,
                    alpha=0.6,
                    label="false positives" if f == 0 else None,
                )
            else:
                idx_active = np.where(results["fields"]["active_trials"][f, :] > 0.5)[0]
                ax_activation.scatter(
                    idx_active,
                    theta_local[idx_active, 0],
                    marker="o",
                    c="tab:green",
                    s=60,
                    alpha=0.6,
                    label="active" if f == 0 else None,
                )

            ax_theta.plot(
                results["fields"]["x"]["theta"][:-1],
                results["fields"]["p_x"]["global"]["theta"][f, ...],
                color=col[f],
            )
            # for i in np.where(active_trials)[0]:
            ax_theta.plot(
                results["fields"]["x"]["theta"][:-1],
                results["fields"]["p_x"]["local"]["theta"][f, active_trials, :].T,
                color=col[f],
                linewidth=0.2,
            )

            # for i in np.where(~active_trials)[0]:
            ax_theta_inactive.plot(
                results["fields"]["x"]["theta"][:-1],
                results["fields"]["p_x"]["local"]["theta"][f, ~active_trials, :].T,
                color=col[f],
                linewidth=0.2,
            )

    ax_activation.legend()
    plt.setp(
        ax_activation,
        xlim=[-1, groundtruth_activation.shape[1]],
        ylim=[0 - 5, nbin + 5],
    )
    plt.setp(ax_theta, xlim=[0, nbin])
    plt.setp(ax_theta_inactive, xlim=[0, nbin])

    x_arr = results["fields"]["x"]["theta"]
    lw = 2
    # print()
    # fig = plt.figure(figsize=(12, 8))

    markercolor = {
        "true positives": "tab:green",
        "false positives": "tab:red",
        "false negatives": "tab:orange",
    }

    ncols = 3
    nrows = max(3, int(np.ceil(n_trials / ncols)))

    gs_trials = fig.add_gridspec(nrows, ncols, left=0.5)
    # plt.setp(gs_trials, xlabel="$\Theta$", ylabel="firing rate")
    # nrows = 4 + int(np.ceil(n_trials / ncols))
    # offset = 4 * ncols
    for trial in range(n_trials):
        # print(trial)
        ax = fig.add_subplot(
            gs_trials[trial // ncols, trial % ncols], sharey=ax if trial > 0 else None
        )
        # nrows, ncols, trial + 1 + offset, sharey=ax if trial > 0 else None
        # )
        ax.plot(results["firingstats"]["trial_map"][trial, :], "k", linewidth=0.5)
        ax.plot(gauss_filter(results["firingstats"]["trial_map"][trial, :], 1), "k")
        ax.set_title(
            f"trial {trial+1}", y=1.0, pad=-14, x=0.3, backgroundcolor="silver"
        )

        if results["fields"]["n_modes"] > 0:

            fields = []
            fields_label = []
            active_trials_1 = results["fields"]["active_trials"][0, :] > 0.5
            isfield = active_trials_1[trial]

            if not comp:
                if isfield:
                    markercolor = "tab:green"
                else:
                    markercolor = None
            elif trial_activation__true_positive[0, trial]:
                markercolor = "tab:green"
            elif trial_activation__false_positive[0, trial]:
                markercolor = "tab:red"
            elif trial_activation__false_negative[0, trial]:
                markercolor = "tab:orange"
            else:
                markercolor = None

            # if isfield:
            fields.append(
                Line2D([], [], color="white", marker="o", markerfacecolor=markercolor)
            )
            fields_label.append("")

            if results["fields"]["n_modes"] > 1:
                active_trials_2 = results["fields"]["active_trials"][1, :] > 0.5
                isfield_2 = active_trials_2[trial]

                if isfield_2:
                    fields.append(
                        Line2D(
                            [],
                            [],
                            color="white",
                            marker="o",
                            markerfacecolor="tab:green",
                        )
                    )
                    fields_label.append("")
            else:
                isfield_2 = False

            ax.legend(
                fields,
                fields_label,
                numpoints=1,
                loc=1,
                frameon=False,
                borderpad=0.1,
                # ncols=2,
            )

            if not isfield and not isfield_2:
                ax.plot(
                    x_arr,
                    get_tuning_curve_trials(
                        x_arr,
                        results["fields"]["parameter"],
                        results["fields"]["n_modes"],
                        None,
                    )[0, trial, :],
                    "r",
                    linewidth=lw,
                )
            if isfield and isfield_2:
                ax.plot(
                    x_arr,
                    get_tuning_curve_trials(
                        x_arr,
                        results["fields"]["parameter"],
                        results["fields"]["n_modes"],
                        fields="all",
                    )[0, trial, :],
                    "r",
                    linewidth=lw,
                )

            if isfield and not isfield_2:
                ax.plot(
                    x_arr,
                    get_tuning_curve_trials(
                        x_arr,
                        results["fields"]["parameter"],
                        results["fields"]["n_modes"],
                        0,
                    )[0, trial, :],
                    "r",
                    linewidth=lw,
                )
            if not isfield and isfield_2:
                ax.plot(
                    x_arr,
                    get_tuning_curve_trials(
                        x_arr,
                        results["fields"]["parameter"],
                        results["fields"]["n_modes"],
                        1,
                    )[0, trial, :],
                    "r",
                    linewidth=lw,
                )
        else:
            ax.plot(
                x_arr,
                get_tuning_curve_trials(
                    x_arr,
                    results["fields"]["parameter"],
                    results["fields"]["n_modes"],
                    None,
                )[0, trial, :],
                "r",
                linewidth=lw,
            )

        ax.spines[["top", "right"]].set_visible(False)
        if trial % ncols > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        if trial // ncols < nrows - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    ax_fmap = fig.add_subplot(gs[0, :])
    ax_fmap.plot(results["firingstats"]["map"], "k-")
    ax_fmap.plot(
        x_arr,
        get_tuning_curve_session(
            x_arr,
            results["fields"]["parameter"],
            results["fields"]["n_modes"],
            "all",
        )[0, :, :].T,
        "r",
        linewidth=2,
        label="inferred model",
    )
    if not (groundtruth_fields is None):
        ax_fmap.plot(
            x_arr,
            np.squeeze(
                model_of_tuning_curve(
                    x_arr[np.newaxis, :],
                    groundtruth_fields,
                    nbin,
                    1,
                    "all",
                )
            ),
            color="tab:green",
            linestyle="--",
            linewidth=2,
            label="groundtruth",
        )
    ax_fmap.legend()
    for axx in [ax_activation, ax_theta, ax_theta_inactive, ax_fmap]:
        axx.spines[["top", "right"]].set_visible(False)

        # for field in groundtruth_fields["PF"]:
        #     ax_fmap.axvline(field["theta"], linestyle="--", color="tab:green")

    A0 = results["fields"]["parameter"]["global"]["A0"][0]
    A = results["fields"]["parameter"]["global"]["A"][:, 0]

    ax.set_ylim([0, max(A0 + A) * 2])
    ax_fmap.set_ylim([0, max(A0 + A) * 2])

    plt.show(block=False)


def get_tuning_curve_session(x, parameter, N_f, fields=None, nbin=40):

    params = {
        "A0": parameter["global"]["A0"][np.newaxis, 0],
        "PF": [],
    }

    for f in range(N_f):
        params["PF"].append({})
        for key in ["A", "sigma", "theta"]:
            params["PF"][f][key] = parameter["global"][key][f, 0][..., np.newaxis]

    n_trials = 1
    return model_of_tuning_curve(
        x[np.newaxis, :],
        params,
        nbin,
        n_trials,
        fields=fields,
    )


def get_tuning_curve_trials(x, parameter, N_f, fields=None, nbin=40):

    params = {
        "A0": parameter["global"]["A0"][np.newaxis, 0],
        "PF": [],
    }

    for f in range(N_f):
        params["PF"].append({})
        for key in ["A", "sigma", "theta"]:

            if not (parameter["local"][key] is None):
                params["PF"][f][key] = parameter["local"][key][f, ..., 0]
            else:
                params["PF"][f][key] = parameter["global"][key][f, 0][..., np.newaxis]

    n_trials = parameter["local"]["theta"].shape[-2]
    return model_of_tuning_curve(
        x[np.newaxis, :],
        params,
        nbin,
        n_trials,
        fields=fields,
    )
