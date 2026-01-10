#!/bin/bash
#SBATCH -a 0-2
#SBATCH -o ./cluster_logs/dataset_gen/%A-%a.o
#SBATCH -e ./cluster_logs/dataset_gen/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J nb201-preds # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source activate python37

# Arrayjob
PYTHONPATH=$PWD python create_pred_dataset_nb201.py \
--seed $SLURM_ARRAY_TASK_ID --dataset $1

# Done
echo "DONE"
echo "Finished at $(date)"
