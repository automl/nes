#!/bin/bash
#SBATCH -a 0-29
#SBATCH -o ./cluster_logs/deepens_amoebanet/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_amoebanet/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J deepens-amoebanet # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source venv/bin/activate

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/train_deepens_baselearner.py --seed_id $SLURM_ARRAY_TASK_ID --working_directory "experiments/fmnist/baselearners/deepens_amoebanet/" --dataset fmnist --num_epochs 15 --scheme deepens_amoebanet --train_amoebanet

# Done
echo "DONE"
echo "Finished at $(date)"
