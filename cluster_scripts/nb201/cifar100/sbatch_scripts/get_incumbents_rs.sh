#!/bin/bash
#SBATCH -a 1-3
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
    --save_dir experiments-nb201/cifar100/outputs/deepens_rs/run_$SLURM_ARRAY_TASK_ID \
    --load_bsls_dir experiments-nb201/cifar100/baselearners/nes_rs/run_$SLURM_ARRAY_TASK_ID \
    --pool_name nes_rs \
    --dataset cifar100

# Done
echo "DONE"
echo "Finished at $(date)"
