#!/bin/bash
#SBATCH -o ./cluster_logs/deepens_rs/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_rs/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J get_incumbents_rs # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source venv/bin/activate

# Arrayjob
PYTHONPATH=$PWD python nes/ensemble_selection/rs_incumbents.py \
    --save_dir experiments/cifar10/outputs/deepens_rs/ \
    --load_bsls_dir experiments/cifar10/baselearners/nes_rs \
    --pool_name nes_rs \
    --dataset cifar10

# Done
echo "DONE"
echo "Finished at $(date)"
