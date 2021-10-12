#!/bin/bash
#SBATCH -o ./cluster_logs/evaluate/%A-%a.o
#SBATCH -e ./cluster_logs/evaluate/%A-%a.e
#SBATCH -p ml_gpu-rtx2080
#SBATCH -a 1
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J ens_from_pool # sets the job name. If not specified, the file name will be used as job name

# Activate virtual environment
#source venv/bin/activate
source activate python36

# mapping from slurm task ID to parameters for python call.
#. cluster_scripts/launcher.config
#IFS=',' grid=( $(eval echo {"${ens_sizes[*]}"}+{"${pools[*]}"}) )
#IFS=' ' read -r -a arr <<< "${grid[*]}"
#IFS=+ read M pool_name <<< "${arr[$SLURM_ARRAY_TASK_ID]}"

if [ "$1" = "nes_rs_esa" ]; then
	PYTHONPATH=$PWD python nes/ensemble_selection/ensembles_from_pools.py \
	    --M "$2" \
	    --pool_name $1 \
	    --save_dir experiments/tiny/ensembles_selected/run_3 \
	    --load_bsls_dir "experiments/tiny/baselearners/$1/run_3" \
	    --dataset tiny
else
	PYTHONPATH=$PWD python nes/ensemble_selection/ensembles_from_pools.py \
	    --M "$2" \
	    --pool_name $1 \
	    --save_dir experiments/tiny/ensembles_selected/run_$SLURM_ARRAY_TASK_ID \
	    --load_bsls_dir "experiments/tiny/baselearners/$1/run_$SLURM_ARRAY_TASK_ID" \
	    --dataset tiny
fi
