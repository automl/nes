#!/bin/bash
#SBATCH -a 0-29
#SBATCH -o ./cluster_logs/deepens_darts/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_darts/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J anch_deepens-darts # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source activate python37

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/train_deepens_baselearner.py \
--seed_id $SLURM_ARRAY_TASK_ID \
--working_directory "experiments/cifar10_low/baselearners/deepens_darts_anchor/" \
--dataset cifar10 --num_epochs 100 --scheme deepens_darts_anchor \
--train_darts --global_seed 1 --batch_size 100 \
--n_layers 8 --init_channels 16 --scheduler cosine \
--anchor --lr 0.025 --wd 0.0 --anch_coeff 0.4
#--anchor --lr 0.03830591641545368 --wd 1.0221751529451768e-05 --anch_coeff 0.4189053283865941 


# .1,- .6

# Done
echo "DONE"
echo "Finished at $(date)"
