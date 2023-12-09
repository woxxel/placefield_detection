#!/bin/bash
#SBATCH -A all
#SBATCH -p medium
#SBATCH -c 12
#SBATCH -t 02:00:00
#SBATCH --mem=8000

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_caiman-1.9.10_py-3.9
source activate caiman-1.9.10_py-3.9

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

cpus=12
dataset=AlzheimerMice_Hayashi
mouse=555wt
session=Session03
path=/scratch/users/$USER/data/$dataset/$mouse/$session

echo $path


python3 ~/program_code/PC_detection/run_script.py $path/OnACID_results.hdf5 $path/aligned_behavior.pkl $path $cpus