#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%A-%a.o
#SBATCH -e ./cluster_logs/evaluate/%A-%a.e
#SBATCH -p bosch_gpu-rtx2080
#SBATCH -a 1-3
#SBATCH --gres=gpu:1  # reserves GPUs

# Activate virtual environment
source activate python37

PYTHONPATH=$PWD python nes/ensemble_selection/ensembles_from_pools.py \
    --M $2 \
    --pool_name $1 \
    --save_dir experiments-nips21/tiny/ensembles_selected/run_$SLURM_ARRAY_TASK_ID \
    --load_bsls_dir "experiments/tiny/baselearners/$1/run_$SLURM_ARRAY_TASK_ID" \
    --dataset tiny \
    --esa $3
