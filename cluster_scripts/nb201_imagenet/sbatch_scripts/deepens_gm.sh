#!/bin/bash
#SBATCH -a 0-2
#SBATCH -o ./cluster_logs/deepens_minima/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_minima/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J deepens-gm # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source venv/bin/activate

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/train_deepens_baselearner.py \
--seed_id $SLURM_ARRAY_TASK_ID \
--working_directory "experiments-nb201/imagenet/baselearners/deepens_minimum/" \
--dataset imagenet --global_seed 1 --nb201 \
--num_epochs 200 --scheme deepens_minimum --train_global_optima

# Done
echo "DONE"
echo "Finished at $(date)"
