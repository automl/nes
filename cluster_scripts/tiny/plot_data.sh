#!/bin/bash

# source config
#. cluster_scripts/launcher.config

# Activate virtual environment
#source activate python36

PYTHONPATH=. python nes/ensemble_selection/make_plot_master.py \
    --Ms 2 3 5 7 10 15 \
    --methods nes_rs nes_re deepens_rs deepens_darts deepens_amoebanet deepens_darts_anchor \
    --save_dir experiments/tiny/outputs/plots-new \
    --load_plotting_data_dir experiments/tiny/outputs/plotting_data \
    --dataset tiny \
    --run run_1 run_2 run_3 run_4 run_5 \
    --plot_type ensemble_size
#PYTHONPATH=. python nes/ensemble_selection/plot_data.py \
    #--Ms 2 3 5 7 10 15 \
    #--methods nes_rs nes_re deepens_rs deepens_darts deepens_amoebanet darts_esa amoebanet_esa nes_rs_esa\
    #--save_dir experiments/tiny/outputs/plots \
    #--load_plotting_data_dir experiments/tiny/outputs/plotting_data \
    #--dataset tiny \
    #--run run_1 run_2 run_3 run_4 run_5
