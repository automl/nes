#!/bin/bash
#SBATCH -o ./cluster_logs/plots/%A-%a.o
#SBATCH -e ./cluster_logs/plots/%A-%a.e
#SBATCH -a 1
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J plotting

# Activate virtual environment
source activate python36

PYTHONPATH=. python nes/ensemble_selection/make_plot_master.py \
     --Ms 2 3 5 7 10 15 \
     --methods nes_rs nes_re deepens_darts darts_hyper darts_rs \
     --save_dir experiments_hyper/cifar$1/outputs/plots \
     --load_plotting_data_dir experiments_hyper/cifar$1/outputs/plotting_data \
     --dataset cifar$1 \
     --run run_1 \
     --plot_type $2


