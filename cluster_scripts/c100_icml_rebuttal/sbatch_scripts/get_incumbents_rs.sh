#!/bin/bash
#SBATCH -o ./cluster_logs/deepens_rs/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_rs/%A-%a.e
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -x dlcgpu37,dlcgpu26
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -a 4-5
#SBATCH -J get_incumbents_rs # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source venv/bin/activate

# Arrayjob
PYTHONPATH=$PWD python nes/ensemble_selection/rs_incumbents.py \
    --save_dir experiments/tiny/outputs/deepens_rs/run_$SLURM_ARRAY_TASK_ID \
    --load_bsls_dir experiments/tiny/baselearners/nes_rs/run_$SLURM_ARRAY_TASK_ID \
    --pool_name nes_rs \
    --dataset tiny

# Done
echo "DONE"
echo "Finished at $(date)"
