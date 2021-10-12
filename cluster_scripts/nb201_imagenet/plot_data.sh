#!/bin/bash

# source config
#. cluster_scripts/launcher.config

# Activate virtual environment
source activate python37

PYTHONPATH=. python nes/ensemble_selection/plot_data_nb201.py \
    --Ms "3" \
    --methods nes_rs deepens_rs nes_re deepens_minimum deepens_gdas \
    --save_dir experiments-nb201/imagenet/outputs/plots \
    --load_plotting_data_dir experiments-nb201/imagenet/outputs/plotting_data \
    --dataset imagenet \
    --run run_1 run_2 run_3
