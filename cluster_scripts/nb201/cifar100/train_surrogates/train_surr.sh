#!/bin/bash
#SBATCH -a 100,200,500
#SBATCH -o ./cluster_logs/nes_bo/%A-%a.o
#SBATCH -e ./cluster_logs/nes_bo/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J nes-bo # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source activate python37

# Arrayjob
PYTHONPATH=$PWD python nes/nasbench201/train_set_trans.py \
--epochs $SLURM_ARRAY_TASK_ID --model_name $1 --num_hidden $2 --save_dir plots3

# Done
echo "DONE"
echo "Finished at $(date)"
