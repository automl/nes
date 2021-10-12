#!/bin/bash
#SBATCH -a 1-5,11-19,21-29,31-39,41-49,51-59,110-120,210-220,310-320,410-420,510-520
#SBATCH -p bosch_gpu-rtx2080
#SBATCH -x dlcgpu15,dlcgpu02,dlcgpu42
#SBATCH -o ./cluster_logs/rand_mutations/%A-%a.o
#SBATCH -e ./cluster_logs/rand_mutations/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J mutations # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION, on gpu $SLURMD_NODENAME"

# Activate virtual environment
source activate python37

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/train_deepens_baselearner.py --arch_id $SLURM_ARRAY_TASK_ID --seed_id $SLURM_ARRAY_TASK_ID --working_directory "experiments-nips21/cifar10/baselearners/random_mutations/" --dataset cifar10 --num_epochs 100 --scheme rs_mutations --arch_path "experiments-nips21/cifar10/baselearners/random_mutations/run_1/random_archs" --global_seed 1

# Done
echo "DONE"
echo "Finished at $(date)"
