#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%x.%A-%a.%N.o
#SBATCH -e ./cluster_logs/evaluate/%x.%A-%a.%N.e
#SBATCH -p bosch_gpu-rtx2080
#SBATCH -a 1-3
#SBATCH --gres=gpu:1  # reserves GPUs

# Activate virtual environment
source activate python37

PYTHONPATH=$PWD python nes/ensemble_selection/evaluate_ensembles.py \
    --M $2 \
    --method $1 \
    --save_dir experiments-nips21/tiny/outputs/plotting_data/run_$SLURM_ARRAY_TASK_ID \
    --nes_rs_bsls_dir experiments/tiny/baselearners/nes_rs/run_$SLURM_ARRAY_TASK_ID \
    --incumbents_dir experiments-nips21/tiny/outputs/deepens_rs/run_$SLURM_ARRAY_TASK_ID/incumbents.txt \
    --load_bsls_dir "experiments/tiny/baselearners/$1/run_$SLURM_ARRAY_TASK_ID" \
    --load_ens_chosen_dir experiments-nips21/tiny/ensembles_selected/run_$SLURM_ARRAY_TASK_ID \
    --dataset tiny \
    --esa $3

