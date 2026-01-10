#!/bin/bash

# Activate virtual environment
source activate python37

PYTHONPATH=. python nes/plots/make_plot_neurips.py --Ms "3" --save_dir experiments/nb201/imagenet/outputs/plots --load_plotting_data_dir experiments/nb201/imagenet/outputs/plotting_data/ --methods nes_re nes_rs --dataset imagenet --runs run_1 run_2 run_3 --plot_type budget
