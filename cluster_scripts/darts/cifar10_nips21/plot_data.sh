#!/bin/bash

# source config
#. cluster_scripts/launcher.config

# Activate virtual environment
#source activate python36

PYTHONPATH=. python nes/ensemble_selection/plot_data_1.py \
    --Ms 2 3 5 7 10 15\
    --methods nes_rs deepens_darts darts_hyper darts_rs joint nes_re \
    --save_dir experiments_hyper/cifar10/outputs/plots \
    --load_plotting_data_dir experiments_hyper/cifar10/outputs/plotting_data \
    --dataset cifar10 \
    --run run_1

