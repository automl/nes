#!/bin/bash
#SBATCH -a 0-2
#SBATCH -o ./cluster_logs/deepens_gdas/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_gdas/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J deepens-gdas # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source activate pt1.3

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/train_deepens_baselearner.py \
--seed_id $SLURM_ARRAY_TASK_ID \
--working_directory "experiments-nb201/cifar10/baselearners/deepens_gdas/" \
--dataset cifar10 --global_seed 1 --nb201 \
--num_epochs 200 --scheme deepens_gdas --train_gdas

# Done
echo "DONE"
echo "Finished at $(date)"
