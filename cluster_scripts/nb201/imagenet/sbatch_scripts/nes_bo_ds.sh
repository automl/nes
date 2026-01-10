#!/bin/bash
#SBATCH -a 1-3
#SBATCH -o ./cluster_logs/nes_bo/%A-%a.o
#SBATCH -e ./cluster_logs/nes_bo/%A-%a.e
#SBATCH -J nes-bo # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source activate python37

# Arrayjob
PYTHONPATH=$PWD python nes/nasbench201/scripts/new_nes_bo.py \
--num_iterations 400 --init_samples 20 --sample_size 3 --B 20 --bo_batch_size 10 \
--working_directory experiments/nb201/imagenet/baselearners/nes_bo --severity_list "0" \
--global_seed $SLURM_ARRAY_TASK_ID --scheme nes_bo --dataset imagenet \
--ls_iters_1 200 --ls_steps_1 0 --ls_iters_2 30 --ls_steps_2 5

# Done
echo "DONE"
echo "Finished at $(date)"
