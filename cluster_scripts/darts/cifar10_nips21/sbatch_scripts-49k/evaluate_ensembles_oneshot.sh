#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%A-%a.o
#SBATCH -e ./cluster_logs/evaluate/%A-%a.e
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -x dlcgpu15,dlcgpu02,dlcgpu42
#SBATCH -a 1-3
#SBATCH --gres=gpu:1  # reserves GPUs

# Activate virtual environment
source activate python37

PYTHONPATH=$PWD python nes/ensemble_selection/evaluate_ensembles.py \
    --M $1 \
    --method nes_rs_oneshot \
    --save_dir experiments-nips21/cifar10/outputs/plotting_data/run_$SLURM_ARRAY_TASK_ID \
    --nes_rs_bsls_dir experiments-nips21/cifar10/baselearners/nes_rs/run_$SLURM_ARRAY_TASK_ID \
    --incumbents_dir experiments-nips21/cifar10/outputs/deepens_rs/run_$SLURM_ARRAY_TASK_ID/incumbents.txt \
    --load_bsls_dir "experiments-nips21/cifar10/baselearners/nes_rs/run_$SLURM_ARRAY_TASK_ID" \
    --load_ens_chosen_dir experiments-nips21/cifar10/ensembles_selected/run_$SLURM_ARRAY_TASK_ID \
    --dataset cifar10 \
    --esa beam_search \
    --arch_id 0
