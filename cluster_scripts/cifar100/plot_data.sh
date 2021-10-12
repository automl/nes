#!/bin/bash

# source config
#. cluster_scripts/launcher.config

# Activate virtual environment
#source activate python36

PYTHONPATH=. python nes/ensemble_selection/make_plot_master.py \
    --Ms 2 3 5 7 10 15 20 30\
    --methods nes_rs deepens_darts deepens_amoebanet darts_rs nes_re deepens_darts_anchor\
    --save_dir experiments/cifar100_low/outputs/plots-new \
    --load_plotting_data_dir experiments/cifar100_low/outputs/plotting_data \
    --dataset cifar100 \
    --run run_1 run_2 run_3 \
    --plot_type ensemble_size
#PYTHONPATH=. python nes/ensemble_selection/plot_data_1.py \
    #--Ms 2 3 5 7 10 15 20 30\
    #--methods nes_rs deepens_darts darts_hyper joint darts_rs nes_re \
    #--save_dir experiments_hyper/cifar100/outputs/plots \
    #--load_plotting_data_dir experiments_hyper/cifar100/outputs/plotting_data \
    #--dataset cifar100 \
    #--run run_1
