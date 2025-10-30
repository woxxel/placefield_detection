import sys
from pathlib import Path

root_dir = Path.cwd().parents[1]
if not root_dir in sys.path:
    sys.path.insert(0, str(root_dir))

# import multiprocessing as mp
# mp.set_start_method("fork", force=True)
# from matplotlib import pyplot as plt

from placefield_detection.utils import prepare_behavior_from_file
from placefield_detection.utils import prepare_activity, load_data

pathMouse = Path("../../../../data/845ad")
pathSession = pathMouse / "Session10"
f = 15.0
nbin = 40
only_active = True

pathBehavior = pathSession / "aligned_behavior.pkl"
behavior = prepare_behavior_from_file(
    pathBehavior,
    only_active=only_active,
    environment_length=120.0,
    nbin=nbin,
    f=15.0,
    T=None,
    calculate_performance=False,
    plt_bool=False,
    plt_trials=False,
)


# plot_behavior(behavior)


pathActivity = [
    file
    for file in pathSession.iterdir()
    if (
        file.stem.startswith("results_CaImAn")
        and not "compare" in file.stem
        and "redetected" in file.stem
    )
][0]

ld = load_data(pathActivity)
n = 0
neuron_activity = ld["S"].copy()
activity = prepare_activity(neuron_activity[n, :], behavior, f=15.0, only_active=True)

from placefield_detection.process_single_neuron import process_single_neuron
from placefield_detection.analyze_results import display_results

process_neuron = process_single_neuron(
    behavior, ["peak", "information", "bayesian"], ["threshold", "bayesian"]
)
results = process_neuron.run_detection(neuron_activity[n, :], show_status=True, nP=12)

display_results(results)
