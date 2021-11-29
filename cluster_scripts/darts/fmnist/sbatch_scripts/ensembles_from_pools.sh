#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%A-%a.o
#SBATCH -e ./cluster_logs/evaluate/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J ens_from_pool # sets the job name. If not specified, the file name will be used as job name
#SBATCH -a 0-5 # should be 2 x (number of ensemble sizes, i.e. length of ens_sizes in launcher.config) - 1

# Activate virtual environment
source venv/bin/activate

# mapping from slurm task ID to parameters for python call.
. cluster_scripts/launcher.config
IFS=',' grid=( $(eval echo {"${ens_sizes[*]}"}+{"${pools[*]}"}) )
IFS=' ' read -r -a arr <<< "${grid[*]}"
IFS=+ read M pool_name <<< "${arr[$SLURM_ARRAY_TASK_ID]}"

PYTHONPATH=$PWD python nes/ensemble_selection/ensembles_from_pools.py \
    --M "$M" \
    --pool_name "$pool_name" \
    --save_dir experiments/fmnist/ensembles_selected/ \
    --load_bsls_dir "experiments/fmnist/baselearners/$pool_name" \
    --dataset fmnist
