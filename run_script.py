import os, sys

from PC_detection.detection import *

_, pathData, pathBehavior, pathResults, cpus = sys.argv
n_processes = int(cpus)

# path_to_data_on_home = os.path.join('/scratch/users',os.environ['USER'],'data',dataset,mouse,session_name)
os.makedirs(pathResults,exist_ok=True)
dPC = PC_detection(pathData,pathBehavior,pathResults,nP=n_processes,plt_bool=False,nbin=40)
dPC.process_session()

print(f"Finished place field detection and stored results to {pathResults}!")
