import numpy as np


def model_of_tuning_curve(x, parameter, n_x, n_trials, fields="all", stacked=False):
    # : int | str | None

    ## build tuning-curve model
    shift = n_x / 2.0
    # parameter["A0"] = np.atleast_1d(parameter["A0"])

    N_in = parameter["A0"].shape[0]
    # print(f"{N_in=}, {n_trials=}, {x.shape=}")

    if not (fields is None) and "fields" in parameter:

        fields = parameter["fields"] if fields == "all" else [parameter["fields"][fields]]
        n_fields = len(fields)

        mean_model = np.zeros((n_fields + 1, N_in, n_trials, x.shape[-1]))
        mean_model[0, ...] = parameter["A0"][..., np.newaxis]

        for f, field in enumerate(fields):
            # for key in field:
            #     field[key] = getattr(field,key)

            mean_model[f + 1, ...] = field.A[..., np.newaxis] * np.exp(
                -(
                    (np.mod(x - field.theta[..., np.newaxis] + shift, n_x) - shift)
                    ** 2
                )
                / (2 * field.sigma[..., np.newaxis] ** 2)
            )
    else:
        mean_model = np.zeros((1, N_in, n_trials, x.shape[-1]))
        mean_model[0, ...] = parameter["A0"][..., np.newaxis]

    if stacked:
        return mean_model
    else:
        return mean_model.sum(axis=0)


def intensity_model_from_position(x, parameter, n_x, fields=None):
    """
    function to build tuning-curve model
    """

    shift = n_x / 2.0
    intensity_model = np.full(len(x), parameter["A0"])

    if not (fields is None) and "fields" in parameter:
        # fields = parameter['PF'] if fields=='all' else [parameter['PF'][fields]]

        for f in fields:
            field = parameter["fields"][f]

            intensity_model += field.A * np.exp(
                -((np.mod(x - field.theta + shift, n_x) - shift) ** 2)
                / (2 * field.sigma ** 2)
            )
    return intensity_model
