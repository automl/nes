#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%A-%a.o
#SBATCH -e ./cluster_logs/evaluate/%A-%a.e
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -a 1-3
#SBATCH --gres=gpu:1  # reserves GPUs

# Activate virtual environment
source activate python37

PYTHONPATH=$PWD python nes/ensemble_selection/ensembles_from_pools.py \
    --M 10 \
    --pool_name $1 \
    --save_dir experiments-rebuttal/cifar100/ensembles_selected/run_$SLURM_ARRAY_TASK_ID \
    --load_bsls_dir "experiments/cifar100_low/baselearners/$1/run_$SLURM_ARRAY_TASK_ID" \
    --dataset cifar100 \
    --validation_size $2
