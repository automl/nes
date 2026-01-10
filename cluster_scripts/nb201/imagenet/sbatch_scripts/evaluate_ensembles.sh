#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%A-%a.o
#SBATCH -e ./cluster_logs/evaluate/%A-%a.e
#SBATCH -J eval_ens # sets the job name. If not specified, the file name will be used as job name
#SBATCH -a 1-3

# Activate virtual environment
source activate python37

PYTHONPATH=$PWD python nes/ensemble_selection/evaluate_ensembles.py \
    --M "3" \
    --method $1 \
    --save_dir experiments/nb201/imagenet/outputs/plotting_data/run_$SLURM_ARRAY_TASK_ID \
    --nes_rs_bsls_dir experiments/nb201/imagenet/baselearners/nes_rs/run_$SLURM_ARRAY_TASK_ID \
    --incumbents_dir experiments/nb201/imagenet/outputs/deepens_rs/run_$SLURM_ARRAY_TASK_ID/incumbents.txt \
    --load_bsls_dir experiments/nb201/imagenet/baselearners/$1/run_$SLURM_ARRAY_TASK_ID \
    --load_ens_chosen_dir experiments/nb201/imagenet/ensembles_selected/run_$SLURM_ARRAY_TASK_ID \
    --dataset imagenet \
    --device -1
