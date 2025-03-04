import logging

from .placecell_detection_methods import (
    peak_method_bare,
    information_method,
    stability_method,
)

from .utils import prepare_activity

from .HierarchicalBayesInference import HierarchicalBayesInference


class process_single_neuron:

    def __init__(
        self,
        behavior,
        parameter,
        mode_place_cell_detection=["peak", "information", "stability", "bayesian"],
        mode_place_field_detection=["bayesian", "threshold"],
        **kwargs,
    ):

        self.behavior = behavior
        self.parameter = parameter

        self.mode_place_cell_detection = mode_place_cell_detection
        self.mode_place_field_detection = mode_place_field_detection

        if "plot_it" in kwargs:
            self.plot_it = kwargs["plot_it"]

    def run_detection(self, activity):

        ## check, if there is enough activity
        # if (activity[self.behavior["active"]] > 0).sum() < 10:
        #     print("Not enough instances of activity detected")
        #     return None

        # t_start = time.time()
        self.place_cell_results = {}
        self.place_cell_detection(activity)
        self.place_field_detection(activity)

        ## finally, gather results
        results = self.place_cell_results["bayesian"]
        results["status"]["is_place_cell"]["peak_method"] = self.place_cell_results[
            "peak"
        ]
        results["status"]["is_place_cell"]["information_method"] = (
            self.place_cell_results["information"]
        )

        return results

    def place_cell_detection(self, activity, **kwargs):

        # self.bin_centers = self.parameter.bin_array_centers

        if "peak" in self.mode_place_cell_detection:
            self.place_cell_results["peak"] = peak_method_bare(
                behavior=self.behavior,
                neuron_activity=activity[self.behavior["active"]],
                nbin=self.parameter.nbin,
                plot=self.plot_it,
                # bin_array_centers=self.nbin_centers,
                **kwargs,
            )

        if "information" in self.mode_place_cell_detection:
            self.place_cell_results["information"] = information_method(
                self.behavior,
                neuron_activity=activity[self.behavior["active"]],
                nbin=self.parameter.nbin,
                plot=self.plot_it,
                # bin_array_centers=self.nbin_centers,
                **kwargs,
            )

        # if "stability" in self.mode_place_cell_detection:
        #     self.place_cell_results["stability"] = stability_method(
        #         self.behavior,
        #         neuron_activity=activity[self.behavior["active"]],
        #         nbin=self.parameter.nbin,
        #         # bin_array_centers=self.nbin_centers,
        #         **kwargs,
        #     )

    def place_field_detection(self, activity, **kwargs):

        nbin = self.parameter.nbin

        if "bayesian" in self.mode_place_field_detection:
            processed_activity = prepare_activity(
                activity, self.behavior["active"], self.behavior["trials"], nbin
            )
            hbm = HierarchicalBayesInference(
                processed_activity["spike_map"],
                self.behavior["trials"]["dwelltime"],
                logLevel=logging.ERROR,
            )

            hbm.model_comparison(hierarchical=["theta"], limit_execution_time=600)

            self.place_cell_results["bayesian"] = hbm.inference_results

            # print("THIS SHOULD BE CHANGED TO PLACE_FIELD_RESULTS")
            # hbm.display_results()
            # handover of results comes here
            # return hbm.inference_results
