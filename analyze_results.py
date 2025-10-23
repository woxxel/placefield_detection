import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .result_structures import (extract_inference_results)
from .utils import gauss_smooth as gauss_filter, model_of_tuning_curve
from .BayesModel import place_field

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

def display_results(
    results,
    idx=None,
    groundtruth_fields=None,
    groundtruth_activation=None,
):

    if not (idx is None):
        results = extract_inference_results(results, idx)


    fields = results["bayesian"]["fields"]

    n_trials, nsteps = fields["p_x"]["local"]["theta"].shape[-2:]
    nbin = 40

    if groundtruth_fields is not None:
        """
        find matching fields between inferred and groundtruth
        """
        # cast to array
        if groundtruth_fields.get("fields"):
            [field.cast_to_array() for field in groundtruth_fields["fields"]]

            field_match = identify_matching_fields(
                fields, groundtruth_fields, nbin=nbin
            )
        else:
            field_match = np.array([], "int")
        
    if not (groundtruth_activation is None):

        trial_activation, (sensitivity, specificity) = calculate_trial_activation(fields["active_trials"], groundtruth_activation, field_match)
        

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, right=0.45)
    # gs_4cols = fig.add_gridspec(6, 4)

    ax_activation = fig.add_subplot(gs[1, :])
    plt.setp(ax_activation, xlabel="trial #", ylabel="$\Theta$")

    ax_theta = fig.add_subplot(gs[2, 0])
    plt.setp(ax_theta, xlabel="$\Theta$", ylabel="p($\Theta$)")

    ax_theta_inactive = fig.add_subplot(gs[2, 1])
    plt.setp(ax_theta_inactive, xlabel="$\Theta$", ylabel="p($\Theta$)")

    comp = groundtruth_fields is not None

    if comp and groundtruth_fields.get("n_fields",0)>0:
        for field in groundtruth_fields["fields"]:
            ax_activation.axhline(field.theta, linestyle="--", color="tab:green")

    col = ["tab:blue", "tab:purple"]

    markercolor = {
        "true_positive": "tab:green",
        "false_positive": "tab:red",
        "false_negative": "tab:orange",
    }

    if fields["n_modes"] > 0:
        for f in range(fields["n_modes"]):
            active_trials = fields["active_trials"][f, :] > 0.5
            theta_local = fields["parameter"]["local"]["theta"][f, ...]

            ax_activation.axhline(
                fields["parameter"]["global"]["theta"][f, 0],
                color="k",
                linestyle="--",
            )

            ax_activation.plot(
                theta_local[:, 0],
                color="r",
                linestyle="-",
                linewidth=0.5,
            )

            if comp:
                for key in ["true_positive", "false_negative", "false_positive"]:
                    idxs = np.where(trial_activation[key][f, ...])[0]

                    ax_activation.scatter(
                        idxs,
                        theta_local[idxs, 0],
                        marker="o",
                        c=markercolor[key],
                        s=60,
                        alpha=0.6,
                        label=key.replace("_", " ") if f == 0 else None,
                    )

            else:
                idx_active = np.where(fields["active_trials"][f, :] > 0.5)[0]
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
                fields["x"]["theta"][:-1],
                fields["p_x"]["global"]["theta"][f, ...],
                color=col[f],
            )
            # for i in np.where(active_trials)[0]:
            ax_theta.plot(
                fields["x"]["theta"][:-1],
                fields["p_x"]["local"]["theta"][f, active_trials, :].T,
                color=col[f],
                linewidth=0.2,
            )

            # for i in np.where(~active_trials)[0]:
            ax_theta_inactive.plot(
                fields["x"]["theta"][:-1],
                fields["p_x"]["local"]["theta"][f, ~active_trials, :].T,
                color=col[f],
                linewidth=0.2,
            )

    ax_activation.legend()
    # results["fields"]["parameter"]["local"]
    plt.setp(
        ax_activation,
        xlim=[-1, fields["parameter"]["local"]["theta"].shape[-2]],
        ylim=[0 - 5, nbin + 5],
    )
    plt.setp(ax_theta, xlim=[0, nbin])
    plt.setp(ax_theta_inactive, xlim=[0, nbin])

    for i,(ax, label) in enumerate(zip([ax_theta, ax_theta_inactive], ["active trials", "inactive trials"])):
        ax.legend(
            [Line2D([], [], color=col[0])],
            [label],
            numpoints=1,
            loc=1,
            frameon=False,
            borderpad=0.1,
        )

    x_arr = fields["x"]["theta"]
    lw = 2

    """
    plot tuning curves for all trials
    """
    ncols = 3
    nrows = max(3, int(np.ceil(n_trials / ncols)))
    gs_trials = fig.add_gridspec(nrows, ncols, left=0.5)

    inferred_parameter = cast_results_to_params(fields["parameter"], fields["n_modes"])
    print(inferred_parameter)
    for trial in range(n_trials):
        ax = fig.add_subplot(
            gs_trials[trial // ncols, trial % ncols], sharey=ax if trial > 0 else None
        )
        # nrows, ncols, trial + 1 + offset, sharey=ax if trial > 0 else None
        # )
        ax.plot(results["firingstats"]["map_trial_rates"][trial, :], "k", linewidth=0.5)
        ax.plot(gauss_filter(results["firingstats"]["map_trial_rates"][trial, :], 1), "k")
        ax.set_title(
            f"trial {trial+1}", y=1.0, pad=-14, x=0.3, backgroundcolor="silver"
        )

        if fields["n_modes"] == 0:
            plot_model(ax, inferred_parameter, None, trial, x_arr, n_trials=n_trials, lw=lw)
        else:

            fields_tmp = []
            fields_label = []

            isfield = np.zeros(fields["n_modes"], "bool")            
            
            if groundtruth_fields is not None:
                for f_gt in range(groundtruth_fields["n_fields"]):
                    
                    # print(f"trial {trial}, f_gt {f_gt}, f {field_match[f_gt]}")
                    f = field_match[f_gt]
                    active_trials = fields["active_trials"][f, :] > 0.5
                    
                    isfield[f] = active_trials[trial] if f>=0 else isfield[f]

                    mcol = None
                    if comp:
                        for key in markercolor:
                            if trial_activation[key][f_gt, trial]:
                                mcol = markercolor[key]
                                break
                    else:
                        if isfield[f]:
                            mcol = "tab:green"
                        else:
                            continue
                    fields_tmp.append(
                        Line2D([], [], color="white", marker="o", markerfacecolor=mcol)
                    )
                    fields_label.append("")
            else:
                for f in range(fields["n_modes"]):
                    active_trials = fields["active_trials"][f, :] > 0.5
                    isfield[f] = active_trials[trial]

                    if isfield[f]:
                        fields_tmp.append(
                            Line2D([], [], color="white", marker="o", markerfacecolor="tab:green")
                        )
                        fields_label.append("")

            ax.legend(
                fields_tmp,
                fields_label,
                numpoints=1,
                loc=1,
                frameon=False,
                borderpad=0.1,
            )
            if ~np.any(isfield):
                plot_model(ax, inferred_parameter, None, trial, x_arr, n_trials=n_trials, lw=lw)
            if np.all(isfield):
                plot_model(ax, inferred_parameter, "all", trial, x_arr, n_trials=n_trials, lw=lw)
            elif np.any(isfield):
                plot_model(ax, inferred_parameter, np.where(isfield)[0][0], trial, x_arr, n_trials=n_trials, lw=lw)


        ax.spines[["top", "right"]].set_visible(False)
        if trial % ncols > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        if trial // ncols < nrows - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    ax_fmap = fig.add_subplot(gs[0, :])
    ax_fmap.plot(results["firingstats"]["map_rates"], "k-")

    inferred_parameter = cast_results_to_params(fields["parameter"], fields["n_modes"],meta=True)
    plot_model(ax_fmap, inferred_parameter, "all", 0, x_arr, n_trials=n_trials, linewidth=2,color="r",label="inferred model")
    if groundtruth_fields is not None:
        plot_model(ax_fmap, groundtruth_fields, "all", 0, x_arr, n_trials=n_trials, linewidth=2,color="tab:green", linestyle="--", label="groundtruth")

    ax_fmap.legend()
    for axx in [ax_activation, ax_theta, ax_theta_inactive, ax_fmap]:
        axx.spines[["top", "right"]].set_visible(False)

    A0 = fields["parameter"]["global"]["A0"][0]
    A = fields["parameter"]["global"]["A"][:, 0]

    max_val = max(A0 + A) * 2 if np.any(A > 0) else max(1, A0 * 10)
    ax.set_ylim([0, max_val])
    ax_fmap.set_ylim([0, max_val])

    plt.show(block=False)


def plot_model(ax,parameter,which,trial,x_arr,n_trials=1,**kwargs):
    nbins = x_arr.max() # == 40
    ax.plot(
        x_arr,
        model_of_tuning_curve(
            x_arr[np.newaxis, :],
            parameter,
            nbins,
            n_trials,
            fields=which,
        )[0, trial, :],
        **{**kwargs, "color": kwargs.get("color","r")},
    )


def identify_matching_fields(fields, groundtruth_fields, nbin=40, print_info=True):
    field_match = np.full(groundtruth_fields["n_fields"], -1, "int")

    theta_gt = np.array([f.theta[0] for f in groundtruth_fields["fields"]])
    theta_inf = fields["parameter"]["global"]["theta"][:, 0]

    dTheta = distance.cdist(theta_gt[..., np.newaxis], theta_inf[...,np.newaxis])

    for f in range(fields["n_modes"]):
        matched_distance = np.nanmin(dTheta[:,f])
        matched_field = np.nanargmin(dTheta[:,f])

        # print(f"field {f}: min distance {matched_distance} at {matched_field}")

        if matched_distance <= 5.0:
            field_match[matched_field] = f

    # print(field_match)
    # print(groundtruth_fields)
    if print_info:
        print(
            f"A0: {groundtruth_fields['A0']} vs {fields['parameter']['global']['A0'][0]}"
        )
        for f_gt in range(groundtruth_fields["n_fields"]):
            for key in ["A", "sigma"]:
                print(
                    f"field {f_gt+1} {key}: {getattr(groundtruth_fields['fields'][f_gt], key)} vs {fields['parameter']['global'][key][field_match[f_gt],0]}"
                )
    return field_match


def calculate_trial_activation(active_trials, active_trials_groundtruth, field_match):

    """
        implement calculating sensitivity and specificity from both fields, if only one is detected that covers both:
        * requires "any" active trial
        * requires checking if fields are overlapping
    """
    _,n_trials = active_trials_groundtruth.shape[:2]
    N_f_gt = len(field_match)
    trial_activation = {
        "true_positive": np.zeros((N_f_gt, n_trials), "bool"),
        "true_negative": np.zeros((N_f_gt, n_trials), "bool"),
        "false_positive": np.zeros((N_f_gt, n_trials), "bool"),
        "false_negative": np.zeros((N_f_gt, n_trials), "bool"),
    }

    join_fields = True
    for f_gt in range(N_f_gt):
        f = field_match[f_gt]
        
        if join_fields:
            active_trials_gt = np.any(active_trials_groundtruth > 0.5,axis=0)
        else:
            active_trials_gt = active_trials_groundtruth[f_gt, :] > 0.5
        if f >= 0:
            active_trials_inf = active_trials[f, :] > 0.5
            trial_activation["true_positive"][f, :] = (
                active_trials_gt & active_trials_inf
            )
            trial_activation["true_negative"][f, :] = (
                ~active_trials_gt & ~active_trials_inf
            )
            trial_activation["false_positive"][f, :] = (
                ~active_trials_gt & active_trials_inf
            )
            trial_activation["false_negative"][f, :] = (
                active_trials_gt & ~active_trials_inf
            )

    sensitivity = trial_activation["true_positive"].sum(axis=1) / (
        trial_activation["true_positive"].sum(axis=1)
        + trial_activation["false_negative"].sum(axis=1)
    )
    specificity = trial_activation["true_negative"].sum(axis=1) / (
        trial_activation["true_negative"].sum(axis=1)
        + trial_activation["false_positive"].sum(axis=1)
    )

    print(f"{sensitivity=}, {specificity=}")

    return trial_activation, (sensitivity, specificity)


def cast_results_to_params(parameter,N_f,meta=False):
    params = {
        "A0": parameter["global"]["A0"][np.newaxis, 0],
        "fields": [],
    }
    for f in range(N_f):
        params_tmp = {}
        for key in ["A", "sigma", "theta"]:

            if parameter["local"].get(key) is None or meta:
                params_tmp[key] = parameter["global"][key][f, 0][..., np.newaxis]
            else:
                params_tmp[key] = parameter["local"][key][f, ..., 0]
        params["fields"].append(place_field(**params_tmp))
    return params
