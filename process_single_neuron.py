import logging

from .alternative_detection_methods import (
    peak_method,
    information_method,
    stability_method,
    thresholding_method,
)

from .utils import prepare_activity
from .analyze_results import build_results

from .BayesModel import HierarchicalBayesInference


class process_single_neuron:

    def __init__(
        self,
        behavior,
        mode_place_cell_detection=["peak", "information"],
        mode_place_field_detection=["bayesian", "threshold"],
        **kwargs,
    ):
        """
        TODO:
        * obtain nbin from behavior
        """
        self.preprocessed = False

        self.behavior = behavior

        self.mode_place_cell_detection = mode_place_cell_detection
        self.mode_place_field_detection = mode_place_field_detection

        self.plot_it = kwargs.get("plot_it", False)

    def run_preprocessing(self, activity):
        """
        Run some general functions on activity, such as analysis of firing activity
        """

        modes = self.mode_place_cell_detection + self.mode_place_field_detection
        unique_modes = list(set(modes))

        self.results = build_results(
            n_cells=1,
            nbin=self.behavior["nbin"],
            n_trials=self.behavior["trials"]["ct"],
            modes=unique_modes,
        )

        self.place_cell_results = {}

        ### throw into separate function

        ## firing rate statistics
        self.prepared_activity = prepare_activity(
            activity,
            self.behavior,
        )

        self.activity = self.prepared_activity["spikes"]
        for key in ["map_rates", "map_trial_rates", "firing_rate"]:
            self.results["firingstats"][key] = self.prepared_activity[key]

        self.preprocessed = True

    def run_detection(self, activity, **kwargs):

        ## check, if there is enough activity
        # if (activity[self.behavior["active"]] > 0).sum() < 10:
        #     print("Not enough instances of activity detected")
        #     return None

        # t_start = time.time()
        self.run_preprocessing(activity)
        assert (
            self.preprocessed
        ), "Preprocessing not run. Please run `run_preprocessing` first."


        modes = self.mode_place_cell_detection + self.mode_place_field_detection
        unique_modes = list(set(modes))
        
        self.place_cell_detection()
        self.place_field_detection(**kwargs)

        ## finally, gather results (appears to carry only bool!)
        # results = self.place_cell_results.get(
        #     "bayesian", {"status": {"is_place_cell": {}}}
        # )
        
        for method in unique_modes:
            if method in self.place_cell_results:
                self.results[method] = self.place_cell_results[method]
            # else:
            # self.results[method] = {"status": {"is_place_cell": False}}
        # self.results["peak"] = self.place_cell_results["peak"]
        # self.results["information"] = self.place_cell_results["information"]

        return self.results

    def place_cell_detection(self, **kwargs):

        if "peak" in self.mode_place_cell_detection:
            self.place_cell_results["peak"] = peak_method(
                behavior=self.behavior,
                neuron_activity=self.activity,
                plot=self.plot_it,
                **kwargs,
            )

        if "information" in self.mode_place_cell_detection:
            self.place_cell_results["information"] = information_method(
                behavior=self.behavior,
                neuron_activity=self.activity,
                plot=self.plot_it,
                **kwargs,
            )

    def place_field_detection(self, **kwargs):

        if "threshold" in self.mode_place_field_detection:
            self.place_cell_results["threshold"] = thresholding_method(
                behavior=self.behavior,
                neuron_activity=self.activity,
                N_f=kwargs.get("N_f", 2),
                **kwargs,
            )

        if "bayesian" in self.mode_place_field_detection:

            hbm = HierarchicalBayesInference(
                self.prepared_activity["map_trial_spikes"],
                self.behavior["trials"]["dwelltime"],
                logLevel=logging.ERROR,
            )

            limit_execution_time = kwargs.get("limit_execution_time", 1200)
            show_status = kwargs.get("show_status", False)
            hbm.model_comparison(
                hierarchical=["theta"],
                limit_execution_time=limit_execution_time,
                show_status=show_status,
            )

            self.place_cell_results["bayesian"] = hbm.inference_results

            # print("THIS SHOULD BE CHANGED TO PLACE_FIELD_RESULTS")
            # hbm.display_results()
            # handover of results comes here
            # return hbm.inference_results
