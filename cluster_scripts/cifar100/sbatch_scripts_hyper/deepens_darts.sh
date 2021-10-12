#!/bin/bash
#SBATCH -a 0-14
#SBATCH -c 4
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -x dlcgpu05,dlcgpu26,dlcgpu37,dlcgpu15
#SBATCH -o ./cluster_logs/deepens_darts/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_darts/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J deepens-darts # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source activate python36

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/darts_rs.py \
--seed_id $SLURM_ARRAY_TASK_ID \
--working_directory "experiments_hyper/cifar100/baselearners/deepens_darts/" \
--dataset cifar100 --num_epochs 100 --scheme deepens_darts --global_seed 1 \
--lr 0.03704869432849922 --wd 0.001842328053779474

# Done
echo "DONE"
echo "Finished at $(date)"
