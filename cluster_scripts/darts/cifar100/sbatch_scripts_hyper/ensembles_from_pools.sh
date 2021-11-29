#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%A-%a.o
#SBATCH -e ./cluster_logs/evaluate/%A-%a.e
#SBATCH -a 1
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J ens_from_pool # sets the job name. If not specified, the file name will be used as job name

# Activate virtual environment
source activate python36

PYTHONPATH=$PWD python nes/ensemble_selection/ensembles_from_pools.py \
    --M "$2" \
    --pool_name $1 \
    --save_dir experiments_hyper/cifar100/ensembles_selected/run_$SLURM_ARRAY_TASK_ID \
    --load_bsls_dir "experiments_hyper/cifar100/baselearners/$1/run_$SLURM_ARRAY_TASK_ID" \
    --dataset cifar100
