#!/bin/bash

# source config
#. cluster_scripts/launcher.config

# Activate virtual environment
source activate pt1.3

#PYTHONPATH=. python nes/ensemble_selection/plot_data.py \
    #--Ms "3" \
    #--methods nes_rs nes_re deepens_rs deepens_gdas deepens_minimum \
    #--save_dir experiments-nb201/cifar100/outputs/plots \
    #--load_plotting_data_dir experiments-nb201/cifar100/outputs/plotting_data \
    #--dataset cifar100 \
    #--run run_1 run_2 run_3

PYTHONPATH=. python nes/plots/make_plot_neurips.py --Ms "3" --save_dir experiments/plots2 --load_plotting_data_dir experiments/nb201/cifar100/outputs/plotting_data/ --methods nes_re nes_bo nes_rs --dataset cifar100 --runs run_1 run_2 run_3 --plot_type budget
