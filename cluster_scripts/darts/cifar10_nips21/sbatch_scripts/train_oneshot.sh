#!/bin/bash
#SBATCH -a 1
#SBATCH -o ./cluster_logs/nes_rs_oneshot/%A-%a.o
#SBATCH -e ./cluster_logs/nes_rs_oneshot/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J rsws # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
#source venv/bin/activate
source activate python37

# Arrayjob
PYTHONPATH=$PWD python nes/randomNAS_release/searchers/random_weight_share.py --save_dir nes/randomNAS_release/oneshot_model

# Done
echo "DONE"
echo "Finished at $(date)"
