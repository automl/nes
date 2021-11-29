#!/bin/bash

# source config
. cluster_scripts/launcher.config

# Activate virtual environment
source venv/bin/activate

PYTHONPATH=. python nes/ensemble_selection/plot_data.py \
    --Ms "${ens_sizes[@]}" \
    --methods nes_rs nes_re deepens_darts deepens_amoebanet deepens_rs \
    --save_dir experiments/fmnist/outputs/plots \
    --load_plotting_data_dir experiments/fmnist/outputs/plotting_data \
    --dataset fmnist
