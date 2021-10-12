#!/bin/bash

# source config
#. cluster_scripts/launcher.config

# Activate virtual environment
#source activate python36

PYTHONPATH=. python nes/ensemble_selection/make_plot_esas.py \
    --Ms 2 3 5 7 10 15 \
    --methods nes_rs nes_re deepens_rs deepens_darts deepens_amoebanet darts_esa amoebanet_esa nes_rs_esa\
    --save_dir experiments-nips21/tiny/outputs/plots/${1} \
    --load_plotting_data_dir experiments-nips21/tiny/outputs/plotting_data \
    --dataset tiny \
    --run run_1 run_2 run_3 \
    --esa $1
