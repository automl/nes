#!/bin/bash	
#SBATCH -a 10 #0-100
#SBATCH -c 4
#SBATCH -o ./cluster_logs/deepens_darts/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_darts/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J anch_hyper # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source activate python37

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/train_deepens_baselearner.py \
--seed_id $1 --arch_id $1 \
--working_directory "experiments-anchor/tiny_2" \
--dataset tiny --num_epochs 100 --scheme deepens_darts \
--n_layers 8 --init_channels 36 --grad_clip --lr 0.1 --scheduler cosine \
--train_darts --global_seed $1 --batch_size 128 --anchor \
--lr 0.025 --wd 0.0 --anch_coeff $2

# Done
echo "DONE"
echo "Finished at $(date)"
