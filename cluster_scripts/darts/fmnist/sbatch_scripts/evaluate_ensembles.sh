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
    --save_dir experiments/fmnist/outputs/plotting_data/ \
    --nes_rs_bsls_dir experiments/fmnist/baselearners/nes_rs/ \
    --incumbents_dir experiments/fmnist/outputs/deepens_rs/incumbents.txt \
    --load_bsls_dir "experiments/fmnist/baselearners/$method" \
    --load_ens_chosen_dir experiments/fmnist/ensembles_selected/ \
    --dataset fmnist
