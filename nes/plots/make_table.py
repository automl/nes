import argparse
import dill as pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

from nes.ensemble_selection.config import BUDGET, PLOT_EVERY

matplotlib.use("Agg")
import os
from pathlib import Path
import numpy as np
import pandas as pd

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = 'dotted'
plt.rcParams['font.size'] = 14


parser = argparse.ArgumentParser()
parser.add_argument(
    "--esa",
    type=str,
    default="beam_search",
    help="Ensemble selection algorithm. See nes/ensemble_selection/esas.py. Default: beam_search.",
)
# parser.add_argument(
#     "--Ms", type=int, nargs="+", help="A sequence of ensemble sizes (M's) to plot.",
# )
parser.add_argument(
    "--save_dir", type=str, help="Directory to save plots.",
)
# parser.add_argument(
#     "--load_plotting_data_dir",
#     type=str,
#     help="Directory where outputs of evaluate_ensembles.py are saved.",
# )
# parser.add_argument(
#     "--methods", type=str, nargs="+", help="A sequence of method names to plot."
# )
# parser.add_argument(
#     "--dataset", choices=["cifar10", "cifar100", "fmnist", "imagenet", "tiny"], type=str, help="Dataset."
# )
# parser.add_argument(
#     "--runs", type=str, default=[''], nargs='+', help="Subdirectories in load_plotting_data_dir over which to average runs."
# )
# parser.add_argument(
#     "--plot_type", type=str, choices=["budget", "ensemble_size", "severity"], help="Which type of plots to make."
# )


args = parser.parse_args()

SAVE_DIR = args.save_dir
data_types = ["val", "test"]
ens_attrs = ["evals", "avg_baselearner_evals", "oracle_evals"]
severities = range(6) #if (args.dataset in ["cifar10", "cifar100", "tiny"]) else range(1)

dataset_to_budget = {
    "cifar10": 400,
    "cifar100": 400, 
    "fmnist": 400,
    "tiny": 200,
    "imagenet": 1000 # TODO: correct?
}

dataset_to_runs = {
    "cifar10": ["run_1"],
    "cifar100": ["run_1"],
}

dataset_to_loading_dir = {
    "cifar10": "/home/zelaa/NIPS20/robust_ensembles/experiments_hyper/cifar10/outputs/plotting_data",
    "cifar100": "/home/zelaa/NIPS20/robust_ensembles/experiments_hyper/cifar100/outputs/plotting_data",
}

# ---------------------------------------------------------------------------- #
#                               Helper functions                               #
# ---------------------------------------------------------------------------- #

def merge_and_fill_trajectories(pandas_data_frames, default_value=None):
    # merge all trajectories keeping all time steps
    df = pd.DataFrame().join(pandas_data_frames, how='outer')

    # forward fill to make it a propper step function
    df = df.fillna(method='ffill')
    if default_value is None:
        # backward fill to replace the NaNs for the early times by the
        # performance of a random configuration
        df = df.fillna(method='bfill')
    else:
        df = df.fillna(default_value)
    return df


def get_trajectories(losses, iterations):
    '''
        methods_dict (dict):
            key (str): method name; should be one in methods
            values (dict): key -> str: 'losses' or 'iterations';
                           values -> list of lists with all the evaluated metrics
    '''
    dfs = []
    for i in range(len(losses)):
        loss = losses[i]
        iteration = iterations[i]
        # print('Run %d, Min: %f'%(i, loss))
        df = pd.DataFrame({str(i): loss}, index=iteration)
        dfs.append(df)

    df = merge_and_fill_trajectories(dfs, default_value=None)
    if df.empty:
        pass

    return np.array(df.index), np.array(df.T)

# # ===================================
# # Get C10 old data
# # ===================================

# if args.dataset == "cifar10":
    


# ===================================
# Plot things
# ===================================

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
plt.rcParams["font.size"] = 14


metric_label = {"loss": "NLL", "error": "Error", "ece": "ECE"}

colors = {
    "nes_rs": "forestgreen",
    "deepens_rs": "dodgerblue",
    "nes_re": "crimson",
    "deepens_darts": "black",
    "deepens_gdas": "cyan",
    "deepens_minimum": "magenta",
    "deepens_amoebanet": "darkorange",
}

markers = {
    'nes_rs': 'v',
    'deepens_rs': 'h',
    'nes_re': 'x',
    'deepens_minimum': '^',
    'deepens_darts': '<',
    'deepens_gdas': '.',
    'deepens_amoebanet': '>'
}
label_names = {
    'nes_rs': 'NES-RS',
    'deepens_rs': 'DeepEns (RS)',
    'nes_re': 'NES-RE',
    'deepens_minimum': 'DeepEns (Global)',
    'deepens_darts': 'DeepEns (DARTS)',
    'deepens_gdas': 'DeepEns (GDAS)',
    'deepens_amoebanet': 'DeepEns (AmoebaNet)',
}

ens_attr_to_title = {
    "evals": "Ensemble",
    "avg_baselearner_evals": "Average baselearner",
    "oracle_evals": "Oracle ensemble",
}

dataset_to_title = {
    "tiny": "Tiny ImageNet",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "fmnist": "FMNIST",
    "imagenet": "ImageNet16",
}

linestyle_method = {
    'deepens_minimum': 'DeepEns (Global)',
    'deepens_darts': 'DeepEns (DARTS)',
    'deepens_gdas': 'DeepEns (GDAS)',
    'deepens_amoebanet': 'DeepEns (AmoebaNet)',
}

methods = ["deepens_darts", "darts_hyper", "darts_rs", "nes_rs", "nes_re"]

table_mean = np.zeros((2*3, 5))
table_err = np.zeros((2*3, 5))
for idx_dataset, dataset in enumerate(["cifar10", "cifar100"]):

    BUDGET = dataset_to_budget[dataset]
    PLOT_EVERY = 25
    runs = dataset_to_runs[dataset]
    load_plot_dir = dataset_to_loading_dir[dataset]
    for data_type in ["test"]:
        for ens_attr in ["evals"]:
            for metric in ["error"]:
                for idx_severity, severity in enumerate([0, 3, 5]):
                    for idx_method, pool_name in enumerate(methods):
                        y_mean = []
                        y_err = []
                        for M in [10]:

                            if pool_name in ["nes_re", "darts_hyper", "darts_rs",
                                            "nes_rs"]:
                                xs = []
                                ys = []

                                for plot_dir in [os.path.join(load_plot_dir, p) for p in runs]:

                                    with open(
                                        os.path.join(
                                            plot_dir,
                                            f"plotting_data__esa_{args.esa}_M_{M}_pool_{pool_name}.pickle",
                                        ),
                                        "rb",
                                    ) as f:
                                        plotting_data = pickle.load(f)

                                    x = plotting_data[str(M)][str(severity)][ens_attr][args.esa][
                                        pool_name
                                    ].x
                                    yy = plotting_data[str(M)][str(severity)][ens_attr][args.esa][
                                        pool_name
                                    ].y
                                    y = [item[data_type][str(severity)][metric] for item in yy]

                                    xs.append(x)
                                    ys.append(y)

                                assert len(xs) == len(ys)
                                assert len(set(xs)) == 1

                                x = xs[0]
                                y = np.array(ys).mean(axis=0)
                                err = np.array(ys).std(axis=0)

                                #label = f"{label_names[pool_name]}"

                                # ax.plot(x, y, label=label, color=colors[pool_name],
                                #         linewidth=2, marker=markers[pool_name],
                                #         markersize=7, markevery=2)
                                # ax.fill_between(x, y - 1.96*err/np.sqrt(len(xs)), y
                                #                 + 1.96*err/np.sqrt(len(xs)),
                                #                 color=colors[pool_name], alpha=0.15)
                                y_mean.append(y[-1])
                                y_err.append(1.96*err[-1]/np.sqrt(len(xs)))

                            elif pool_name in ["deepens_darts", "deepens_pcdarts",
                                            "deepens_amoebanet", "deepens_gdas",
                                            "deepens_minimum"]:
                                if runs != ['']:
                                    with open(
                                        os.path.join(
                                            load_plot_dir,
                                            runs[0],
                                            f"plotting_data__M_{M}_pool_{pool_name}.pickle",
                                        ),
                                        "rb",
                                    ) as f:
                                        plotting_data = pickle.load(f)
                                else:
                                    with open(
                                        os.path.join(
                                            load_plot_dir,
                                            f"plotting_data__M_{M}_pool_{pool_name}.pickle",
                                        ),
                                        "rb",
                                    ) as f:
                                        plotting_data = pickle.load(f)

                                x = plotting_data[str(M)][str(severity)][ens_attr][pool_name].x
                                yy = plotting_data[str(M)][str(severity)][ens_attr][pool_name].y
                                y = yy[data_type][str(severity)][metric]

                                label = f"{label_names[pool_name]}"

                                # ax.axhline(
                                #     y,
                                #     label=label,
                                #     linestyle="--",
                                #     color=colors[pool_name],
                                #     linewidth=2,
                                # )

                                y_mean.append(y)
                                y_err.append(0) # don't have error bars for these

                            elif pool_name in ["deepens_rs"]:
                                xs = []
                                ys = []

                                for plot_dir in [os.path.join(load_plot_dir, p) for p in runs]:
                                    with open(
                                        os.path.join(
                                            plot_dir,
                                            f"plotting_data__M_{M}_pool_{pool_name}.pickle",
                                        ),
                                        "rb",
                                    ) as f:
                                        plotting_data = pickle.load(f)

                                    x = plotting_data[str(M)][str(severity)][ens_attr][pool_name].x
                                    yy = plotting_data[str(M)][str(severity)][ens_attr][pool_name].y
                                    y = [item[data_type][str(severity)][metric] for item in yy]

                                    # extend line until end of plot.
                                    x = x + [BUDGET]
                                    y = y + [y[-1]]

                                    xs.append(x)
                                    ys.append(y)

                                x, all_pools = get_trajectories(ys, xs)

                                std = all_pools.std(axis=0)
                                mean = np.mean(all_pools, axis=0)

                                label = f"{label_names[pool_name]}"

                                # ax.plot(x, mean, label=label, color=colors[pool_name],
                                #         linewidth=2, marker=markers[pool_name],
                                #         markersize=7, markevery=2)
                                # ax.fill_between(x, mean -
                                #                 1.96*std/np.sqrt(len(xs)), mean
                                #                 + 1.96*std/np.sqrt(len(xs)),
                                #                 color=colors[pool_name], alpha=0.15)

                                y_mean.append(mean[-1])
                                y_err.append(1.96*std[-1]/np.sqrt(len(xs)))


                        y_mean = np.array(y_mean)
                        y_err = np.array(y_err)

                        table_mean[(3*idx_dataset + idx_severity), idx_method] = y_mean[0]
                        table_err[(3*idx_dataset + idx_severity), idx_method] = y_err[0]


table_mean = 100 * table_mean
table_err = 100 * table_err

table = [
    [None for _ in range(5)] for _ in range(6)
]

for row in range(6):
    for col in range(5):

        thres = min(table_mean[row]) + table_err[row][np.argmin(table_mean[row])]

        if table_mean[row][col] <= thres:
            latex_str = f"$\\mathbf{{ {table_mean[row][col]:.1f} }}$"
        else:
            latex_str = f"${table_mean[row][col]:.1f}$"
        if table_err[row][col] != 0:
            latex_str = latex_str + f"\\tiny{{$\pm {table_err[row][col]:.1f} $}}"

        table[row][col] = latex_str

table = pd.DataFrame(table)
table.to_csv("experiments_hyper/table_hyper.csv", header=False, index=False)

