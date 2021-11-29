#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%A-%a.o
#SBATCH -e ./cluster_logs/evaluate/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J eval_ens # sets the job name. If not specified, the file name will be used as job name
#SBATCH -a 0-14 # should be 5 x (number of ensemble sizes, i.e. length of ens_sizes in launcher.config) - 1

# Activate virtual environment
source venv/bin/activate

# mapping from slurm task ID to parameters for python call.
. cluster_scripts/launcher.config
IFS=',' grid=( $(eval echo {"${ens_sizes[*]}"}+{"${methods[*]}"}) )
IFS=' ' read -r -a arr <<< "${grid[*]}"
IFS=+ read M method <<< "${arr[$SLURM_ARRAY_TASK_ID]}"

PYTHONPATH=$PWD python nes/ensemble_selection/evaluate_ensembles.py \
    --M "$M" \
    --method "$method" \
    --save_dir experiments/cifar10/outputs/plotting_data/ \
    --nes_rs_bsls_dir experiments/cifar10/baselearners/nes_rs/ \
    --incumbents_dir experiments/cifar10/outputs/deepens_rs/incumbents.txt \
    --load_bsls_dir "experiments/cifar10/baselearners/$method" \
    --load_ens_chosen_dir experiments/cifar10/ensembles_selected/ \
    --dataset cifar10
