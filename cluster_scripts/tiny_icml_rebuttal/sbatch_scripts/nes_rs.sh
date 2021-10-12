#!/bin/bash
#SBATCH -a 0-399
#SBATCH -c 4
#SBATCH -o ./cluster_logs/nes_rs/%A-%a.o
#SBATCH -e ./cluster_logs/nes_rs/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J nes-rs # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION, on gpu $SLURMD_NODENAME"

# Activate virtual environment
source activate python37

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/run_nes_rs.py --working_directory=experiments-nips21/tiny/baselearners/nes_rs --arch_id $SLURM_ARRAY_TASK_ID --seed_id $SLURM_ARRAY_TASK_ID --dataset tiny --num_epochs 100 --batch_size 128 --n_layers 8 --init_channels 36 --grad_clip --lr 0.1 --scheduler cosine --global_seed $1

# Done
echo "DONE"
echo "Finished at $(date)"
