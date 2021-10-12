#!/bin/bash
#SBATCH -a 1-3
#SBATCH -o ./cluster_logs/nes_re/%A-%a.o
#SBATCH -e ./cluster_logs/nes_re/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J nb201-nes-re # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source venv/bin/activate

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/run_nes_re_nb201.py \
--num_iterations 400 --num_epochs 200 --population_size 50 --sample_size 10 \
--working_directory experiments-nb201/cifar10/baselearners/nes_re --severity_list "0 5" \
--global_seed $SLURM_ARRAY_TASK_ID --scheme nes_re --dataset cifar10 --nb201

# Done
echo "DONE"
echo "Finished at $(date)"
