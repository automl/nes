#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%A-%a.o
#SBATCH -e ./cluster_logs/evaluate/%A-%a.e
#SBATCH -a 1
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J eval_ens # sets the job name. If not specified, the file name will be used as job name

# Activate virtual environment
source activate python36

PYTHONPATH=$PWD python nes/ensemble_selection/evaluate_ensembles.py \
    --M "$2" \
    --method "$1" \
    --save_dir experiments_hyper/cifar100/outputs/plotting_data/run_$SLURM_ARRAY_TASK_ID \
    --nes_rs_bsls_dir experiments_hyper/cifar100/baselearners/nes_rs/run_$SLURM_ARRAY_TASK_ID \
    --load_bsls_dir "experiments_hyper/cifar100/baselearners/$1/run_$SLURM_ARRAY_TASK_ID" \
    --load_ens_chosen_dir experiments_hyper/cifar100/ensembles_selected/run_$SLURM_ARRAY_TASK_ID \
    --dataset cifar100


