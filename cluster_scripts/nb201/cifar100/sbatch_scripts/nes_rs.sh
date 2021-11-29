#!/bin/bash
#SBATCH -a 0-399
#SBATCH -o ./cluster_logs/nes_rs/%A-%a.o
#SBATCH -e ./cluster_logs/nes_rs/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J nes-rs # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source venv/bin/activate

# Arrayjob
PYTHONPATH=$PWD python nes/nasbench201/scripts/run_nes_rs.py \
--working_directory=experiments/nb201/cifar100/baselearners/nes_rs --arch_id \
$SLURM_ARRAY_TASK_ID --seed_id $SLURM_ARRAY_TASK_ID --dataset cifar100 \
--global_seed $1

# Done
echo "DONE"
echo "Finished at $(date)"
