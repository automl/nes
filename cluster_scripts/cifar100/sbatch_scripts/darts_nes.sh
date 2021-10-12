#!/bin/bash
#SBATCH -a 0-100
#SBATCH -c 4
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -x dlcgpu05,dlcgpu26,dlcgpu37,dlcgpu15
#SBATCH -o ./cluster_logs/darts_rs/%A-%a.o
#SBATCH -e ./cluster_logs/darts_rs/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J darts-rs # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source activate python36

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/darts_rs.py \
--arch_id $SLURM_ARRAY_TASK_ID --seed_id $SLURM_ARRAY_TASK_ID \
--working_directory "experiments/cifar100/baselearners/darts_rs/" \
--dataset cifar100 --num_epochs 100 --scheme darts_rs \
--global_seed 1 --nes


# Done
echo "DONE"
echo "Finished at $(date)"
