import os, sys, shutil

from detection import *

print(sys.argv)
_, datapath, dataset, mouse, session, cpus = sys.argv
n_processes = int(cpus)

path = os.path.join(datapath,dataset,mouse,session)

pathData = os.path.join(path,'OnACID_results.hdf5')
pathBehavior = os.path.join(path,'aligned_behavior.pkl')
pathResults = path
# $path_on_cluster/OnACID_results.hdf5 $path_on_cluster/aligned_behavior.pkl $path_on_cluster $cpus
# path=/scratch/users/$USER/data/$dataset/$mouse/$session
# path_on_cluster=os.path.join('/usr/users',os.environ['USER'],'/data/',dataset,mouse,session)


# n_processes = 0

os.makedirs(pathResults,exist_ok=True)
dPC = PC_detection(pathData,pathBehavior,pathResults,nP=n_processes,plt_bool=False,nbin=100)
dPC.process_session()

print(f"Finished place field detection and stored results to {pathResults}!")
path_to_data_on_home = os.path.join('/scratch/users',os.environ['USER'],f'data/{dataset}/{mouse}/{session}')

print(f'Attempt to copy data towards {path_to_data_on_home}')
os.makedirs(path_to_data_on_home,exist_ok=True)
shutil.copyfile(os.path.join(pathResults,'PC_fields.pkl'),os.path.join(path_to_data_on_home,'PC_fields.pkl'))
