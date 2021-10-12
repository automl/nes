#!/bin/bash
#SBATCH -a 1-20
#SBATCH -o ./cluster_logs/nes_re/%A-%a.o
#SBATCH -e ./cluster_logs/nes_re/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J nes-re # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source activate python36

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/darts_re.py --array_id \
$SLURM_ARRAY_TASK_ID --total_num_workers=20 \
--num_iterations 400 --num_epochs 100 --population_size 50 --sample_size 10 \
--nic_name eth0 --working_directory experiments_hyper/cifar100/baselearners/nes_re \
--global_seed 1 --scheme nes_re --dataset cifar100 \
--lr 0.03704869432849922 --wd 0.001842328053779474

# Done
echo "DONE"
echo "Finished at $(date)"
