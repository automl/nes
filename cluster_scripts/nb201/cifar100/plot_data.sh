#!/bin/bash

# source config
#. cluster_scripts/launcher.config

# Activate virtual environment
source activate pt1.3

PYTHONPATH=. python nes/ensemble_selection/plot_data.py \
    --Ms "3" \
    --methods nes_rs nes_re deepens_rs deepens_gdas deepens_minimum \
    --save_dir experiments-nb201/cifar100/outputs/plots \
    --load_plotting_data_dir experiments-nb201/cifar100/outputs/plotting_data \
    --dataset cifar100 \
    --run run_1 run_2 run_3
