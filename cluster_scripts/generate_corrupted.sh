#!/bin/bash
#
# submit to the right queue
#SBATCH --gres gpu:1
#SBATCH -a 0-18
#
# redirect the output/error to some files
#SBATCH -o ./cluster_logs/corruption_logs/%A-%a.o
#SBATCH -e ./cluster_logs/corruption_logs/%A-%a.e
#

source venv/bin/activate
PYTHONPATH=$PWD python data/generate_corrupted.py $SLURM_ARRAY_TASK_ID
