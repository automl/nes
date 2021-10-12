#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%x.%A-%a.%N.o
#SBATCH -e ./cluster_logs/evaluate/%x.%A-%a.%N.e
#SBATCH -p bosch_gpu-rtx2080 #alldlc_gpu-rtx2080
#SBATCH --gres=gpu:1  # reserves GPUs

# Activate virtual environment
source activate python37

# mapping from slurm task ID to parameters for python call.
#. cluster_scripts/launcher.config
#IFS=',' grid=( $(eval echo {"${ens_sizes[*]}"}+{"${methods[*]}"}) )
#IFS=' ' read -r -a arr <<< "${grid[*]}"
#IFS=+ read M method <<< "${arr[$SLURM_ARRAY_TASK_ID]}"

PYTHONPATH=$PWD python nes/ensemble_selection/evaluate_ensembles.py \
    --M "$2" \
    --method "$1" \
    --save_dir experiments/cifar100_low/outputs/plotting_data/run_$SLURM_ARRAY_TASK_ID \
    --nes_rs_bsls_dir experiments/cifar100_low/baselearners/nes_rs/run_$SLURM_ARRAY_TASK_ID \
    --incumbents_dir experiments/cifar100_low/outputs/deepens_rs/run_$SLURM_ARRAY_TASK_ID/incumbents.txt \
    --load_bsls_dir "experiments/cifar100_low/baselearners/$1/run_$SLURM_ARRAY_TASK_ID" \
    --load_ens_chosen_dir experiments/cifar100_low/ensembles_selected/run_$SLURM_ARRAY_TASK_ID \
    --dataset cifar100


