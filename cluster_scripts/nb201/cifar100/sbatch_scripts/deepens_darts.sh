#!/bin/bash
#SBATCH -a 0-2
#SBATCH -o ./cluster_logs/deepens_darts/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_darts/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J deepens-darts # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source venv/bin/activate

# Arrayjob
PYTHONPATH=$PWD python nes/nasbench201/scripts/train_deepens_baselearner.py \
--seed_id $SLURM_ARRAY_TASK_ID \
--working_directory "experiments/nb201/cifar100/baselearners/deepens_darts/" \
--dataset cifar100 --scheme deepens_darts --train_darts --global_seed 1

# Done
echo "DONE"
echo "Finished at $(date)"
