import os, time, math, warnings, pickle
from pathlib import Path

# from caiman.utils.utils import
import multiprocessing as mp

from matplotlib import pyplot as plt
import numpy as np

import logging

from .utils import (
    detection_parameters,
    get_reliability,
    save_data,
    get_spiking_data,
    # build_struct_PC_results,
    prepare_behavior_from_file,
    prepare_activity,
    load_dict_from_hdf5,
)
from .process_single_neuron import process_single_neuron

from .alternative_detection_methods import stability_method

# from .HierarchicalBayesInference import (
#     build_inference_results,
# )

from .analyze_results import build_results, handover_inference_results

# from utils.utils_analysis import prepare_quantiles

# from placefield_detection_inference import *
# from unbiased_MI.estimate_unbiased_information import estimate_unbiased_information

warnings.filterwarnings("ignore")

logger = logging.getLogger("ultranest")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)


class process_session:

    def __init__(self, plot_it=False, save_it=False, suffix="", **kwargs):
        """
        initializes the placefield_detection class

        loads parameters from detection_parameters class, which
        determine some of the meta-parameters of the different methods

        TODO:
            * enable providing behavior data, instead of loading/processing it automatically
            * change method into 2-level class:
                1. general class, containing all kind of data
                2. class containing only data and functions, relevant or single place field detection
            * check: is across-trial noise similar in each neuron? able to actually run it via HBM?
            * put plotting methods in extra file
            * make sure to have consistent input variables
        """

        ### set global parameters and data for all processes to access
        self.options = {
            "plot_it": plot_it,
            "plot_theory": False,  # same
            "plot_save": False,  # same
            "activity": "calcium",  # data provided: 'calcium' or 'spikes'
            "info": "I_mutual",  # 'I_mutual', 'I_per_sec', 'I_per_spike' // "unbiased"?
        }

        self.parameter = detection_parameters(**kwargs)

        # self.process_session()

    def from_file(
        self,
        path_data,
        path_behavior,
        path_results=None,
        mode_place_cell_detection=["peak", "information", "stability", "bayesian"],
        mode_place_field_detection=["bayesian", "threshold"],
        specific_n=None,
        suffix="",
        nP=1,
    ):
        # self.parameter.set_paths(path_data, path_results, suffix=suffix)

        # if Path(path_results).exists():
        #     print(f"results file {path_results} already exists")
        #     return

        behavior = prepare_behavior_from_file(
            path_behavior, nbin=self.parameter.nbin, f=self.parameter.f
        )  ## load and process behavior

        neuron_activity, neuron_quality = self.load_neuron_data(path_data)

        results = self.from_input(
            behavior,
            neuron_activity,
            mode_place_cell_detection=mode_place_cell_detection,
            mode_place_field_detection=mode_place_field_detection,
            specific_n=specific_n,
            nP=nP,
        )

        # return results

        # ## handover data from detection (needed? is contained in other data as well)
        # n_processed = self.n_cells if specific_n is None else len(specific_n)
        # for n in range(n_processed):
        #     results["status"]["SNR"][n] = neuron_quality["SNR_comp"][n]
        #     results["status"]["r_value"][n] = neuron_quality["r_values"][n]

        if path_results is None:
            return results
        # else:
        #     print(f"store results as {path_results}")
        #     save_data(results, path_results)

    def from_input(
        self,
        behavior,
        neuron_activity,
        specific_n=None,
        mode_place_cell_detection=["peak", "information", "stability", "bayesian"],
        mode_place_field_detection=["bayesian", "threshold"],
        path_results=None,
        nP=1,
        **kwargs,
    ):
        """
        preprocesses data to feed into parallelized place field detection

        Input parameters:

        """

        self.n_cells = neuron_activity.shape[0]

        t_start = time.time()
        self.process_single_neuron = process_single_neuron(
            behavior,
            mode_place_cell_detection,
            mode_place_field_detection,
            plot_it=self.options["plot_it"],
        )

        if isinstance(specific_n, (int, list, np.ndarray)):
            if len(specific_n) == 1:
                nP = 0
            idx_process = np.atleast_1d(specific_n)
        else:
            idx_process = np.arange(self.n_cells)

        n_cells_process = len(idx_process)

        print("run detection on %d neurons" % n_cells_process)

        ### first, run batch processing (such as stability method, unbiased MI, ...)
        results_batch = {}
        if "stability" in self.process_single_neuron.mode_place_cell_detection:
            print("run stability method")
            results_batch["stability"] = stability_method(
                behavior,
                neuron_activity[:, behavior["active"]],
                idx_process,
            )

        ### then, run single neuron processing in parallel
        results_tmp = []
        if nP > 1:
            batchSz = 5 * nP
            nBatch = n_cells_process // batchSz

            # progress = tqdm.tqdm(range(nBatch + 1))

            with mp.Pool(nP) as pool:
                # pool = get_context("spawn").Pool(self.para['nP'])
                for i in range(nBatch + 1):
                    idx_batch = idx_process[
                        i * batchSz : min(n_cells_process, (i + 1) * batchSz)
                    ]
                    results_tmp.extend(
                        pool.map(
                            self.process_single_neuron.run_detection,
                            neuron_activity[idx_batch, :],
                        )
                    )
                    # progress.set_description(
                    # print(
                    #     f"\t --- {min(n_cells_process, (i + 1) * batchSz)} / {n_cells_process} neurons processed\t --- \t time passed: {(time.time() - t_start):5.2f}s"
                    # )
        else:
            for n0, n in enumerate(idx_process):
                results_tmp.append(
                    self.process_single_neuron.run_detection(neuron_activity[n, :])
                )
                print(
                    f"\t\t\t ------ {n0 + 1} / {n_cells_process} neurons processed\t ------ \t time passed: {(time.time() - t_start):7.2f}s"
                )

        # handover results
        modes = mode_place_cell_detection + mode_place_field_detection
        unique_modes = list(set(modes))

        # return results_tmp

        results = build_results(
            n_cells=len(results_tmp),
            nbin=behavior["nbin"],
            n_trials=behavior["trials"]["ct"],
            N_f=2,
            hierarchical=["theta"],
            modes=unique_modes,
        )

        for n, res in enumerate(results_tmp):
            if res:
                results = handover_inference_results(
                    res, results, n, excluded_keys=["x"]
                )

        results["stability"] = results_batch["stability"]

        if path_results is None:
            return results
        else:
            print(f"store results as {path_results}")
            save_data(results, path_results)

        return results

    # def process_session_old(
    #     self,
    #     S=None,
    #     f_max=1,
    #     specific_n=None,
    #     #   dataSet='OnACID_results.hdf5',
    #     #   artificial=False,
    #     return_results=False,
    #     rerun=False,
    #     mode_info="MI",
    #     mode_MI="BAE",
    #     mode_activity="spikes",
    #     assignment=None,
    # ):

    #     global t_start
    #     t_start = time.time()

    #     self.f_max = f_max
    #     self.para["modes"]["info"] = mode_info
    #     self.para["modes"]["activity"] = mode_activity
    #     self.tmp = {}  ## dict to store some temporary variables in

    #     # if (result.status['SNR'] < 2) | (results['r_value'] < 0):
    #     #     print('component not considered to be proper neuron')
    #     #     return

    #     calc_MI = False
    #     MI = np.full(self.n_cells, np.nan)

    #     if calc_MI:
    #         self.unbiased_info = estimate_unbiased_information(
    #             self.neuron_activity["qtl"], (self.behavior["binpos"] / 5).astype("int")
    #         )

    #         active_ind = self.unbiased_info["firing_statistics"][
    #             "sufficiently_active_cells_indexes"
    #         ]
    #         sig_tuned_ind = self.unbiased_info["firing_statistics"][
    #             "significantly_tuned_and_active_cells_indexes"
    #         ]

    #         bool_sig_tuned = [i in sig_tuned_ind for i in range(self.n_cells)]

    #         if mode_MI == "BAE":
    #             MI[active_ind] = self.unbiased_info["information"]["MI_BAE"].flatten()

    #         elif mode_MI == "SSR":
    #             MI[active_ind] = self.unbiased_info["information"]["MI_SSR"].flatten()
    #     else:
    #         bool_sig_tuned = np.full(self.n_cells, True)

    #     self.bool_sig_tuned = bool_sig_tuned
    #     self.MI = MI

    #     if isinstance(specific_n, "int"):
    #         # self.S = S[specific_n,:]
    #         self.para["n"] = specific_n
    #         # self.process_single_neuron.run_detection(self.neuron_activity['S'][specific_n,:])
    #         self.process_single_neuron.run_detection(
    #             self.neuron_activity["S"][specific_n, :],
    #             bool_sig_tuned[specific_n],
    #             MI[specific_n],
    #         )
    #         return

    #     # if rerun:
    #     #     if artificial:
    #     #         #nDat,pathData = get_nPaths(self.para['pathSession'],'artificialData_analyzed_n')
    #     #         #for i in range(nDat):

    #     #         f = open(self.para['svname_art'],'rb')
    #     #         PC_processed = pickle.load(f)
    #     #         f.close()

    #     #         #PC_processed = extend_dict(PC_processed,ld_tmp['fields']['parameter'].shape[0],ld_tmp)
    #     #     else:
    #     #         PC_processed = {}
    #     #         PC_processed['status'] = loadmat(os.path.join(self.para['pathSession'],os.path.splitext(self.para['svname_status'])[0]+'.pkl'),squeeze_me=True)
    #     #         PC_processed['fields'] = loadmat(os.path.join(self.para['pathSession'],os.path.splitext(self.para['svname_fields'])[0]+'.pkl'),squeeze_me=True)
    #     #         PC_processed['firingstats'] = loadmat(os.path.join(self.para['pathSession'],os.path.splitext(self.para['svname_firingstats'])[0]+'.pkl'),squeeze_me=True)

    #     #     self.para['modes']['info'] = False

    #     # idx_process = np.where(np.isnan(PC_processed['status']['Bayes_factor'][:,0]))[0]
    #     #     idx_process = np.where(PC_processed['status']['MI_value']<=0.1)[0]
    #     #     print(idx_process)
    #     # elif not (assignment is None):
    #     #     idx_process = assignment
    #     #     print(idx_process)
    #     # else:

    #     if pf_detection == "threshold":
    #         pf_method = threshold
    #     elif pf_detection == "bayesian":
    #         pf_method = bayesian

    #     idx_process = np.arange(self.n_cells)
    #     n_cells_process = len(idx_process)

    #     # print('in class para:',self.process_single_neuron.para)
    #     # if n_cells_process:

    #     print("run detection on %d neurons" % n_cells_process)

    #     result_tmp = []
    #     if self.para["nP"] > 0:
    #         batchSz = 500
    #         nBatch = n_cells_process // batchSz
    #         with mp.Pool(self.para["nP"]) as pool:
    #             # pool = get_context("spawn").Pool(self.para['nP'])

    #             for i in range(nBatch + 1):
    #                 idx_batch = idx_process[
    #                     i * batchSz : min(n_cells_process, (i + 1) * batchSz)
    #                 ]
    #                 # result_tmp.extend(pool.starmap(self.process_single_neuron.run_detection,zip(self.neuron_activity['S'][idx_batch,:])))
    #                 result_tmp.extend(
    #                     pool.starmap(
    #                         self.process_single_neuron.run_detection,
    #                         zip(
    #                             self.neuron_activity["S"][idx_batch, :],
    #                             bool_sig_tuned[idx_batch],
    #                             MI[idx_batch],
    #                         ),
    #                     )
    #                 )
    #                 print(
    #                     "\n\t\t\t ------ %d / %d neurons processed\t ------ \t time passed: %7.2fs\n"
    #                     % (
    #                         min(n_cells_process, (i + 1) * batchSz),
    #                         n_cells_process,
    #                         time.time() - t_start,
    #                     )
    #                 )
    #     else:
    #         for n0, n in enumerate(idx_process):
    #             # result_tmp.append(self.process_single_neuron.run_detection(self.neuron_activity['S'][n,:]))
    #             result_tmp.append(
    #                 self.process_single_neuron.run_detection(
    #                     self.neuron_activity["S"][n, :], bool_sig_tuned[n], MI[n]
    #                 )
    #             )
    #             print(
    #                 "\t\t\t ------ %d / %d neurons processed\t ------ \t time passed: %7.2fs"
    #                 % (n0 + 1, n_cells_process, time.time() - t_start)
    #             )

    #     ## can't this be done more efficiently?
    #     results = build_struct_PC_results(
    #         self.n_cells,
    #         self.para["nbin"],
    #         self.behavior["trials"]["ct"],
    #         1 + len(self.para["CI_arr"]),
    #     )

    #     # return result_tmp,results
    #     for n in range(self.n_cells):
    #         for key_type in result_tmp[0].keys():
    #             for key in result_tmp[0][key_type].keys():
    #                 if key[0] == "_":
    #                     continue
    #                 if rerun:
    #                     if n in idx_process:
    #                         n0 = np.where(idx_process == n)[0][0]
    #                         results[key_type][key][n, ...] = result_tmp[n0][key_type][
    #                             key
    #                         ]
    #                     else:
    #                         # ((~np.isnan(PC_processed['status']['Bayes_factor'][n,0])) | (key in ['MI_value','MI_p_value','MI_z_score','Isec_value','Isec_p_value','Isec_z_score'])):# | (n>=idx_process[10])):
    #                         results[key_type][key][n, ...] = PC_processed[key_type][
    #                             key
    #                         ][n, ...]
    #                 elif not (assignment is None):
    #                     if n < len(idx_process):
    #                         n0 = idx_process[n]
    #                         results[key_type][key][n0, ...] = result_tmp[n][key_type][
    #                             key
    #                         ]
    #                 else:
    #                     n0 = np.where(idx_process == n)[0][0]
    #                     results[key_type][key][n, ...] = result_tmp[n0][key_type][key]

    #     print("time passed (overall): %7.2f" % (time.time() - t_start))

    #     if self.para["plt_bool"]:
    #         self.plt_results(result_tmp[0])

    #     if return_results:
    #         return results
    #     else:
    #         print("saving results...")
    #         with open(self.para["pathResults"], "wb") as f_open:
    #             pickle.dump(results, f_open)

    #         return

    def load_neuron_data(self, path_data):

        ## load activity data from CaImAn results
        ld = load_dict_from_hdf5(path_data)

        # activity = {}
        # activity["C"] = ld["C"]
        neuron_activity = ld["S"]
        neuron_activity[neuron_activity < 0] = 0

        if (
            neuron_activity.shape[0] > 8000
        ):  ## check, whether array is of proper shape - threshold might need to be adjusted for other data
            neuron_activity = neuron_activity.transpose()
        # self.neuron_activity = activity
        # activity['S'] = gauss_filter(neuron_activity['S'],2,axis=1)

        n_cells = neuron_activity.shape[0]

        neuron_quality = {}
        for key in ["SNR_comp", "r_values"]:
            neuron_quality[key] = (
                ld[key] if key in ld.keys() else np.full(n_cells, np.NaN)
            )

        neuron_quality["idx_evaluate"] = np.logical_and(
            neuron_quality["SNR_comp"] > self.parameter.SNR_thr,
            neuron_quality["r_values"] > self.parameter.r_value_thr,
        )

        # for key in ['idx_evaluate','idx_previous']:
        # neuron_activity[key] = ld[key].astype('bool') if key in ld.keys() else np.full(self.n_cells,np.NaN)

        return neuron_activity, neuron_quality

    # def prepare_activity(self):

    # self.neuron_activity['S_active'] = S[:,self.behavior['active']]

    def calc_Icorr(self, S, trials_S):

        S /= S[S > 0].mean()
        lag = [0, self.para["f"] * 2]
        nlag = lag[1] - lag[0]
        T = S.shape[0]

        print("check if range is properly covered in C_cross (and PSTH)")
        print(
            "speed, speed, speed!! - what can be vectorized? how to get rid of loops?"
        )
        print("spike train generation: generate a single, long one at once (or smth)")
        print(
            "how to properly generate surrogate data? single trials? conjoint trials? if i put in a rate only, sums will even out to homogenous process for N large"
        )

        PSTH = np.zeros((self.para["nbin"], nlag))
        C_cross = np.zeros((self.para["nbin"], nlag))
        for x in range(self.para["nbin"]):
            for t in range(self.behavior["trials"]["ct"]):
                idx_x = np.where(self.behavior["trials"]["trial"][t]["binpos"] == x)[0]
                # print(idx_x)
                if len(idx_x):
                    i = (
                        self.behavior["trials"]["frame"][t] + idx_x[0]
                    )  ## find entry to position x in trial t
                    # print('first occurence of x=%d in trial %d (start: %d) at frame %d'%(x,t,self.behavior['trials']['frame'][t],i))
                    PSTH[x, : min(nlag, T - (i + lag[0]))] += S[
                        i + lag[0] : min(T, i + lag[1])
                    ]
            for i in range(1, nlag):
                C_cross[x, i] = np.corrcoef(PSTH[x, :-i], PSTH[x, i:])[0, 1]
            C_cross[np.isnan(C_cross)] = 0
            # C_cross[x,:] = np.fft.fft(C_cross[x,:])
        # PSTH /= nlag/self.para['f']*self.behavior['trials']['ct']
        fC_cross = np.fft.fft(C_cross)

        rate = PSTH.sum(1) / (nlag * self.behavior["trials"]["ct"])
        # print(rate)

        Icorr = np.zeros(self.para["nbin"])
        Icorr_art = np.zeros(self.para["nbin"])
        Icorr_art_std = np.zeros(self.para["nbin"])

        for x in range(self.para["nbin"]):
            print(x)
            Icorr[x] = (
                -1
                / 2
                * rate[x]
                * np.log2(1 - fC_cross[x, :] / (rate[x] + fC_cross[x, :])).sum()
            )
            Icorr_art[x], Icorr_art_std[x] = self.calc_Icorr_data(rate[x], nlag)

        plt.figure()
        plt.plot(Icorr)
        plt.errorbar(range(self.para["nbin"]), Icorr_art, yerr=Icorr_art_std)
        # plt.plot(Icorr_art,'r')
        plt.show(block=False)

        return PSTH, C_cross, Icorr

    # self.behavior['trials']['frame'][t]

    def calc_Icorr_data(self, rate, T, N_bs=10):

        t = np.linspace(0, T - 1, T)
        Icorr = np.zeros(N_bs)

        nGen = int(math.ceil(1.1 * T * rate))
        u = np.random.rand(
            N_bs, nGen
        )  ## generate random variables to cover the whole time
        t_AP = np.cumsum(
            -(1 / rate) * np.log(u), 1
        )  ## generate points of homogeneous pp
        # print(t_AP)
        for L in range(N_bs):
            t_AP_now = t_AP[L, t_AP[L, :] < T]
            idx_AP = np.argmin(np.abs(t_AP_now[:, np.newaxis] - t[np.newaxis, :]), 1)

            PSTH = np.zeros(T)
            for AP in idx_AP:
                PSTH[AP] += 1

            # C_cross = np.correlate(PSTH,PSTH)
            # print(C_cross)
            C_cross = np.zeros(T)
            for i in range(1, T):
                C_cross[i] = np.corrcoef(PSTH[:-i], PSTH[i:])[0, 1]
            C_cross[np.isnan(C_cross)] = 0
            fC_cross = np.fft.fft(C_cross)
            Icorr[L] = -1 / 2 * rate * np.log2(1 - fC_cross / (rate + fC_cross)).sum()

        return Icorr.mean(), Icorr.std()

    def plt_data(
        self,
        n,
        S=None,
        results=None,
        ground_truth=None,
        activity_mode="calcium",
        sv=False,
        suffix="",
    ):
        print("plot data")
        rc("font", size=10)
        rc("axes", labelsize=12)
        rc("xtick", labelsize=8)
        rc("ytick", labelsize=8)

        highlight_trial = False
        if (S is None) and ~hasattr(self, "S"):
            # pathDat = os.path.join(self.para['pathSession'],'results_redetect.mat')
            pathDat = os.path.join(
                self.para["pathSession"], f"OnACID_results{suffix}.hdf5"
            )
            ld = loadmat(pathDat, variable_names=["S", "C"])
            self.S = ld["S"]
            self.C = ld["C"]
            S_raw = self.S[n, :]
            C = self.C[n, :]
        else:
            S_raw = np.squeeze(S[n, :].toarray())
            C = np.squeeze(S[n, :].toarray())
        # print(S_raw.shape)

        S_thr, rate, _ = get_spiking_data(
            S_raw[self.behavior["active"]], f=self.para["f"], sd_r=self.para["Ca_thr"]
        )

        if activity_mode == "spikes":
            S = S_thr  # np.floor(S_raw / S_thr).astype("float")
        else:
            S = C

        # if ~hasattr(self,'results'):
        self.results = {}
        if results is None:
            print("reimplement loading results")

        else:
            self.results["status"] = results["status"]
            self.results["fields"] = results["fields"]
            self.results["firingstats"] = results["firingstats"]

        t_start = 0  # 200
        t_end = 600  # 470
        n_trial = 12

        fig = plt.figure(figsize=(7, 3), dpi=150)
        if ground_truth is None:
            ax_Ca = plt.axes([0.1, 0.7, 0.6, 0.25])
            ax_dwell = plt.axes([0.7, 0.9, 0.2, 0.075])
            ax_loc = plt.axes([0.1, 0.1, 0.6, 0.6])
            ax1 = plt.axes([0.7, 0.1, 0.25, 0.5])
        else:
            ax_Ca = plt.axes([0.125, 0.75, 0.7, 0.225])
            ax_trial_act = plt.axes([0.125, 0.65, 0.7, 0.1])
            ax_loc = plt.axes([0.125, 0.15, 0.7, 0.5])
            ax1 = plt.axes([0.825, 0.15, 0.15, 0.5])
        # ax2 = plt.axes([0.6,0.275,0.35,0.175])
        # ax3 = plt.axes([0.1,0.08,0.4,0.25])
        # ax4 = plt.axes([0.6,0.08,0.35,0.175])

        idx_longrun = self.behavior["active"]
        t_longrun = self.behavior["time"][idx_longrun]
        t_stop = self.behavior["time"][~idx_longrun]
        ax_Ca.bar(
            t_stop,
            np.ones(len(t_stop)) * 1.2 * S.max(),
            color=[0.9, 0.9, 0.9],
            zorder=0,
        )

        ax_Ca.plot(self.behavior["time"], C, "k", linewidth=0.2)
        ax_Ca.plot(self.behavior["time"], S_raw, "r", linewidth=0.3)
        ax_Ca.plot([0, self.behavior["time"][-1]], [S_thr, S_thr])
        ax_Ca.set_ylim([0, 1.2 * S_raw.max()])
        ax_Ca.set_xlim([t_start, t_end])
        ax_Ca.set_xticks([])
        ax_Ca.set_ylabel("Ca$^{2+}$")
        ax_Ca.set_yticks([])

        if not (ground_truth is None):
            trials_frame = np.where(np.diff(self.behavior["binpos"]) < -10)[0] + 1
            trial_act = np.zeros(self.behavior["binpos"].shape + (2,))
            f = np.where(~np.isnan(self.results["fields"]["parameter"][n, :, 3, 0]))[0][
                0
            ]
            ff = np.where(
                np.abs(
                    self.results["fields"]["parameter"][n, f, 3, 0]
                    - ground_truth["theta"][n, ...]
                )
                < 5
            )[0]
            rel, _, trial_field = get_reliability(
                self.results["firingstats"]["trial_map"][n, ...],
                self.results["firingstats"]["map"][n, :],
                self.results["fields"]["parameter"][n, ...],
                f,
            )
            for i in range(len(trials_frame) - 1):
                trial_act[trials_frame[i] : trials_frame[i + 1], 0] = ground_truth[
                    "activation"
                ][n, ff, i]
                trial_act[trials_frame[i] : trials_frame[i + 1], 1] = trial_field[
                    i
                ]  # self.results['firingstats']['trial_field'][n,f,i]

            ax_trial_act.bar(
                self.behavior["time"], trial_act[:, 0], facecolor=[0, 1, 0], bottom=1
            )
            ax_trial_act.bar(
                self.behavior["time"],
                trial_act[:, 1],
                facecolor=[0.4, 1, 0.4],
                bottom=0,
            )
            ax_trial_act.plot([t_start, t_end], [1, 1], color=[0.5, 0.5, 0.5], lw=0.3)
            ax_trial_act.text(t_end - 75, 0.15, "detected", fontsize=8)
            ax_trial_act.text(t_end - 100, 1.15, "ground truth", fontsize=8)
            ax_trial_act.set_ylim([0, 2])
            ax_trial_act.set_xlim([t_start, t_end])
            ax_trial_act.set_xticks([])
            ax_trial_act.set_yticks([])

        ax_loc.plot(
            self.behavior["time"],
            self.behavior["binpos"],
            ".",
            color=[0.6, 0.6, 0.6],
            zorder=5,
            markeredgewidth=0,
            markersize=1,
        )
        idx_active = (S > 0) & self.behavior["active"]
        idx_inactive = (S > 0) & ~self.behavior["active"]

        t_active = self.behavior["time"][idx_active]
        pos_active = self.behavior["binpos"][idx_active]
        S_active = S[idx_active]

        t_inactive = self.behavior["time"][idx_inactive]
        pos_inactive = self.behavior["binpos"][idx_inactive]
        S_inactive = S[idx_inactive]
        if activity_mode == "spikes":
            ax_loc.scatter(t_active, pos_active, s=S_active, c="r", zorder=10)
            ax_loc.scatter(t_inactive, pos_inactive, s=S_inactive, c="k", zorder=10)
        else:
            ax_loc.scatter(
                t_active,
                pos_active,
                s=(S_active / S.max()) ** 2 * 10 + 0.1,
                color="r",
                zorder=10,
            )
            ax_loc.scatter(
                t_inactive,
                pos_inactive,
                s=(S_inactive / S.max()) ** 2 * 10 + 0.1,
                color="k",
                zorder=10,
            )
        ax_loc.bar(
            t_stop,
            np.ones(len(t_stop)) * self.para["L_track"],
            color=[0.9, 0.9, 0.9],
            zorder=0,
        )
        if highlight_trial:
            ax_loc.fill_between(
                [
                    self.behavior["trials"]["t"][n_trial],
                    self.behavior["trials"]["t"][n_trial + 1],
                ],
                [0, 0],
                [self.para["L_track"], self.para["L_track"]],
                color=[0, 0, 1, 0.2],
                zorder=1,
            )
            ax_Ca.fill_between(
                [
                    self.behavior["trials"]["t"][n_trial],
                    self.behavior["trials"]["t"][n_trial + 1],
                ],
                [0, 0],
                [1.2 * S_raw.max(), 1.2 * S_raw.max()],
                color=[0, 0, 1, 0.2],
                zorder=1,
            )
        ax_loc.set_ylim([0, self.para["L_track"]])
        ax_loc.set_xlim([t_start, t_end])
        ax_loc.set_xlabel("t [s]")
        ax_loc.set_ylabel("Location [bin]")

        if ground_truth is None:
            ax_dwell.bar(
                self.para["bin_array"], self.behavior["trials"]["dwelltime"].sum(0)
            )

        # fmap = self.get_firingmap(S[self.behavior['active']],self.behavior['binpos'][self.behavior['active']],self.behavior['trials']['dwelltime'].sum(0))
        bin_array = np.linspace(0, self.para["L_track"], self.para["nbin"])
        # ax1.barh(bin_array,fmap,facecolor='r',alpha=0.5,height=self.para['L_track']/self.para['nbin'],label='$\\bar{\\nu}$')

        # fmap = self.results['firingstats']['map'][n,:]
        # fmap_norm = np.nansum(self.results['firingstats']['map'][n,:])
        ax1.barh(
            bin_array,
            self.results["firingstats"]["map"][n, :],
            height=self.para["L_track"] / self.para["nbin"],
            facecolor="r",
            alpha=0.5,
            label="(all)",
        )

        trials_frame = np.where(np.diff(self.behavior["binpos"]) < -10)[0] + 1
        active = np.copy(self.behavior["active"])
        rel, _, trial_field = get_reliability(
            self.results["firingstats"]["trial_map"][n, ...],
            self.results["firingstats"]["map"][n, :],
            self.results["fields"]["parameter"][n, ...],
            f,
        )
        for i in range(len(trials_frame) - 1):
            active[trials_frame[i] : trials_frame[i + 1]] &= trial_field[
                i
            ]  # self.results['firingstats']['trial_field'][n,f,i]
        fmap = self.get_firingmap(S[active], self.behavior["binpos"][active])
        ax1.barh(
            bin_array,
            fmap,
            facecolor="b",
            alpha=0.5,
            height=self.para["L_track"] / self.para["nbin"],
            label="(active)",
        )
        # ax1.legend(fontsize=10,bbox_to_anchor=[0.1,1.1],loc='lower left')
        ax1.set_xlabel("$\\bar{\\nu}$")
        # ax1.barh(self.para['bin_array'][i],fr_mu[i],facecolor='b',height=1)
        # ax1.errorbar(self.results['firingstats']['map'][n,:],bin_array,xerr=self.results['firingstats']['CI'][n,...],ecolor='r',linewidth=0.2,linestyle='',fmt='',label='1 SD confidence')

        ##flierprops = dict(marker='.', markerfacecolor='k', markersize=0.5)
        ##h_bp = ax1.boxplot(Y,flierprops=flierprops)#,positions=self.para['bin_array'])
        ax1.set_yticks([])  # np.linspace(0,100,6))
        ##ax1.set_yticklabels(np.linspace(0,100,6).astype('int'))
        ax1.set_ylim([0, self.para["L_track"]])
        ax1.set_xlim([0, np.nanmax(self.results["firingstats"]["map"][n, :]) * 1.2])
        ax1.set_xticks([])
        ##ax1.set_xlabel('Ca$^{2+}$-event rate $\\nu$')
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ##ax1.set_ylabel('Position on track')
        # ax1.legend(title='# trials = %d'%self.behavior['trials']['ct'],loc='lower left',bbox_to_anchor=[0.55,0.025],fontsize=8)#[h_bp['boxes'][0]],['trial data'],
        plt.tight_layout()
        plt.show(block=False)
        if sv:
            pathSv = os.path.join(
                self.para["pathFigs"],
                "PC_detection_example_%s.png" % self.paths["suffix"],
            )
            plt.savefig(pathSv)
            print("Figure saved @ %s" % pathSv)

    def plot_activity(self, n, activity_mode="calcium", sv=False, suffix=""):

        n_trial = 4

        fig = plt.figure(figsize=(7, 5), dpi=150)
        ax_Ca = plt.axes([0.1, 0.7, 0.7, 0.25])
        # add_number(fig,ax_Ca,order=1)
        ax_loc = plt.axes([0.1, 0.15, 0.7, 0.55])
        ax1 = plt.axes([0.8, 0.15, 0.175, 0.55])

        time = self.behavior["time_raw"]
        C = self.neuron_activity["C"][n, :]

        # S_raw = self.S

        neuron_activity = prepare_activity(
            self.neuron_activity["S"][n, :],
            self.behavior["active"],
            self.behavior["trials"],
            nbin=self.para["nbin"],
            f=self.para["f"],
        )

        # firingmap = self.get_and_plot_firingmap(n)
        # firingstats = get_firingstats_from_trials(
        #     neuron_activity["trial_map"], self.behavior["trials"]["dwelltime"], N_bs=100
        # )

        # S_raw = self.neuron_activity['S'][n,:]
        S = neuron_activity["S"]
        # if activity_mode == 'spikes':
        #     print('spikes')
        #     S = S_raw>S_thr
        # else:
        #     S = S_raw

        idx_longrun = self.behavior["active"]

        # time = np.linspace(0,600,len(S))
        # t_longrun = time[idx_longrun]
        t_stop = time[~idx_longrun]
        ax_Ca.bar(
            t_stop,
            np.ones(len(t_stop)) * 1.2 * S.max(),
            color=[0.9, 0.9, 0.9],
            zorder=0,
        )

        ax_Ca.fill_between(
            [
                self.behavior["time"][self.behavior["trials"]["start"][n_trial]],
                self.behavior["time"][self.behavior["trials"]["start"][n_trial + 1]],
            ],
            [0, 0],
            [1.2 * S.max(), 1.2 * S.max()],
            color=[0, 0, 1, 0.2],
            zorder=1,
        )

        ax_Ca.plot(time, C, "k", linewidth=0.2)
        ax_Ca.plot(time, S, "r", linewidth=1)
        # ax_Ca.plot(time[[0-1]],[S_thr,S_thr])
        ax_Ca.set_ylim([0, 1.2 * S.max()])
        ax_Ca.set_xlim(time[[0, -1]])
        ax_Ca.set_xticks([])
        ax_Ca.set_ylabel("Ca$^{2+}$")
        ax_Ca.set_yticks([])

        ax_loc.plot(
            time,
            self.behavior["binpos_raw"],
            ".",
            color="k",
            zorder=5,
            markeredgewidth=0,
            markersize=1.5,
        )
        idx_active = (S > 0) & self.behavior["active"]
        idx_inactive = (S > 0) & ~self.behavior["active"]

        t_active = time[idx_active]
        pos_active = self.behavior["binpos_raw"][idx_active]
        S_active = S[idx_active]

        t_inactive = time[idx_inactive]
        pos_inactive = self.behavior["binpos_raw"][idx_inactive]
        S_inactive = S[idx_inactive]
        # if self.para['modes']['activity'] == 'spikes':
        #     ax_loc.scatter(t_active,pos_active,s=3,color='r',zorder=10)
        #     ax_loc.scatter(t_inactive,pos_inactive,s=3,color='k',zorder=10)
        # else:
        # ax_loc.scatter(t_active,pos_active,s=(S_active/S.max())**2*10+0.1,color='r',zorder=10)
        # ax_loc.scatter(t_inactive,pos_inactive,s=(S_inactive/S.max())**2*10+0.1,color='k',zorder=10)

        ax_loc.scatter(
            t_active, pos_active, s=np.minimum(8, S_active + 2), color="r", zorder=10
        )
        ax_loc.scatter(
            t_inactive,
            pos_inactive,
            s=np.minimum(8, S_inactive + 2),
            color="k",
            zorder=10,
        )

        ax_loc.bar(
            t_stop,
            np.ones(len(t_stop)) * self.para["L_track"],
            width=1 / 15,
            color=[0.9, 0.9, 0.9],
            zorder=0,
        )
        ax_loc.fill_between(
            [
                self.behavior["time"][self.behavior["trials"]["start"][n_trial]],
                self.behavior["time"][self.behavior["trials"]["start"][n_trial + 1]],
            ],
            [0, 0],
            [self.para["nbin"], self.para["nbin"]],
            color=[0, 0, 1, 0.2],
            zorder=1,
        )

        ax_loc.set_ylim([0, self.para["nbin"]])
        ax_loc.set_xlim(time[[0, -1]])
        ax_loc.set_xlabel("time [s]")
        ax_loc.set_ylabel("position [bins]")

        i = 25
        ax1.barh(
            self.para["bin_array"],
            firingstats["map"],
            facecolor="b",
            alpha=0.2,
            height=1,
            label="$\\bar{\\nu}$",
        )
        ax1.barh(
            self.para["bin_array"][i], firingstats["map"][i], facecolor="b", height=1
        )

        ax1.errorbar(
            firingstats["map"],
            self.para["bin_array"],
            xerr=firingstats["CI"],
            ecolor="r",
            linewidth=0.2,
            linestyle="",
            fmt="",
            label="1 SD confidence",
        )
        # Y = trials_fmap/self.behavior['trials']['dwelltime']
        # mask = ~np.isnan(Y)
        # Y = [y[m] for y, m in zip(Y.T, mask.T)]

        # #flierprops = dict(marker='.', markerfacecolor='k', markersize=0.5)
        # #h_bp = ax1.boxplot(Y,flierprops=flierprops)#,positions=self.para['bin_array'])
        ax1.set_yticks([])  # np.linspace(0,100,6))
        # ax1.set_yticklabels(np.linspace(0,100,6).astype('int'))
        ax1.set_ylim([0, self.para["nbin"]])

        # ax1.set_xlim([0,np.nanmax(fr_mu[np.isfinite(fr_mu)])*1.2])
        # ax1.set_xticks([])
        # ax1.set_xlabel('Ca$^{2+}$-event rate $\\nu$')
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        # ax1.set_ylabel('Position on track')
        ax1.legend(
            title="# trials = %d" % self.behavior["trials"]["ct"],
            loc="lower left",
            bbox_to_anchor=[0.55, 0.025],
            fontsize=8,
        )  # [h_bp['boxes'][0]],['trial data'],

        return

    def get_and_plot_firingmap(self, n):

        # spikeNr,_,_ = get_spikeNr(self.neuron_activity['S'][n,:])
        neuron_activity = prepare_activity(
            self.neuron_activity["S"][n, :],
            self.behavior["active"],
            self.behavior["trials"],
            nbin=self.para["nbin"],
            f=self.para["f"],
        )
        # return neuron_activity

        # firingstats = get_firingstats_from_trials(
        #     neuron_activity["trial_map"], self.behavior["trials"]["dwelltime"], N_bs=100
        # )

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.plot(self.behavior["time_raw"], neuron_activity["S"], "r", linewidth=0.3)

        ax = fig.add_subplot(122)
        ax.bar(
            self.para["bin_array"],
            firingstats["map"],
            facecolor="b",
            width=1,
            alpha=0.2,
        )
        # ax.bar(self.para['bin_array'],fmap,facecolor='r',width=1,alpha=0.2)
        ax.errorbar(
            self.para["bin_array"],
            firingstats["map"],
            firingstats["CI"],
            ecolor="r",
            linestyle="",
            fmt="",
            elinewidth=0.3,
        )

        plt.show(block=False)

        return firingstats


#### ---------------- end of class definition -----------------
