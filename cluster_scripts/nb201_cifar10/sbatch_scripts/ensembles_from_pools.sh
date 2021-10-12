#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%A-%a.o
#SBATCH -e ./cluster_logs/evaluate/%A-%a.e
#SBATCH -a 1-3
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J ens_from_pool # sets the job name. If not specified, the file name will be used as job name

# Activate virtual environment
source venv/bin/activate

# mapping from slurm task ID to parameters for python call.
#. cluster_scripts/launcher.config
#IFS=',' grid=( $(eval echo {"${ens_sizes[*]}"}+{"${pools[*]}"}) )
#IFS=' ' read -r -a arr <<< "${grid[*]}"
#IFS=+ read M pool_name <<< "${arr[$SLURM_ARRAY_TASK_ID]}"

PYTHONPATH=$PWD python nes/ensemble_selection/ensembles_from_pools.py \
    --M "3" \
    --pool_name $1 \
    --save_dir experiments-nb201/cifar10/ensembles_selected/run_$SLURM_ARRAY_TASK_ID \
    --load_bsls_dir "experiments-nb201/cifar10/baselearners/$1/run_$SLURM_ARRAY_TASK_ID" \
    --dataset cifar10 --device -1

