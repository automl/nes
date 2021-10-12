#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -x dlcgpu15,dlcgpu02,dlcgpu42
#SBATCH -o ./cluster_logs/deepens_darts_50k/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_darts_50k/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J nes_rs_50k # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
#source venv/bin/activate
source activate python37

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/train_deepens_baselearner.py --arch_id $1 --seed_id $1 --working_directory "experiments-nips21/cifar10/baselearners/nes_rs_50k/" --dataset cifar10 --num_epochs 80 --scheme nes_rs_50k --arch_path "experiments-nips21/cifar10/baselearners/nes_rs/run_${2}/random_archs" --global_seed $2 --full_train

# Done
echo "DONE"
echo "Finished at $(date)"
