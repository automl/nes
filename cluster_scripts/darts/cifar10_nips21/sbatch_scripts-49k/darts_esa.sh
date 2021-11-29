#!/bin/bash
#SBATCH -a 146-199
#SBATCH -o ./cluster_logs/deepens_darts/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_darts/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J darts-esa # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
#source venv/bin/activate
source activate python37

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/train_deepens_baselearner.py --seed_id $SLURM_ARRAY_TASK_ID --working_directory "experiments-nips21/cifar10/baselearners/darts_esa/" --dataset cifar10 --num_epochs 100 --scheme darts_esa --train_darts

# Done
echo "DONE"
echo "Finished at $(date)"
