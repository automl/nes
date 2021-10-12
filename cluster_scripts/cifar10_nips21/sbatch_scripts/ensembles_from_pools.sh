#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%A-%a.o
#SBATCH -e ./cluster_logs/evaluate/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -x dlcgpu15,dlcgpu02,dlcgpu42
#SBATCH -a 1-3
#SBATCH -J ens_from_pool # sets the job name. If not specified, the file name will be used as job name

# Activate virtual environment
source activate python37
# conda activate python37

PYTHONPATH=$PWD python nes/ensemble_selection/ensembles_from_pools.py \
    --M $2 \
    --pool_name $1 \
    --save_dir experiments-nips21/cifar10/ensembles_selected/run_$SLURM_ARRAY_TASK_ID \
    --load_bsls_dir experiments-nips21/cifar10/baselearners/$1/run_$SLURM_ARRAY_TASK_ID \
    --dataset cifar10 \
    --esa $3 \
    --arch_id 0 \
    --diversity_strength $4 # used only for esa = beam_search_with_div
