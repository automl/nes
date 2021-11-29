#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -x dlcgpu05,dlcgpu26,dlcgpu37,dlcgpu15
#SBATCH -a 0-399
#SBATCH -c 4
#SBATCH -o ./cluster_logs/deepens_darts/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_darts/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J darts-esa # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source venv/bin/activate

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/train_deepens_baselearner.py \
--seed_id $SLURM_ARRAY_TASK_ID \
--working_directory "experiments/tiny/baselearners/darts_esa/" \
--dataset tiny --num_epochs 100 --scheme darts_esa \
--train_darts --global_seed 1 --batch_size 128 \
--n_layers 8 --init_channels 36 --grad_clip --lr 0.1 --scheduler cosine


# Done
echo "DONE"
echo "Finished at $(date)"
