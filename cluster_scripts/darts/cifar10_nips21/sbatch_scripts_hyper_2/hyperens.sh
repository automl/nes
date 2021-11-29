#!/bin/bash
#SBATCH -a 0-199
#SBATCH -c 4
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -x dlcgpu05,dlcgpu26,dlcgpu37,dlcgpu15
#SBATCH -o ./cluster_logs/darts_hyper/%A-%a.o
#SBATCH -e ./cluster_logs/darts_hyper/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J hyperens # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source activate python36

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/darts_rs_2.py \
--arch_id $SLURM_ARRAY_TASK_ID --seed_id $SLURM_ARRAY_TASK_ID \
--working_directory "experiments_hyper_2/cifar10/baselearners/darts_hyper/" \
--dataset cifar10 --num_epochs 100 --scheme darts_hyper \
--global_seed 1 --hyperensemble

# Done
echo "DONE"
echo "Finished at $(date)"
