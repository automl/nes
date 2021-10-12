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
parser.add_argument(
    "--Ms", type=int, nargs="+", help="A sequence of ensemble sizes (M's) to plot.",
)
parser.add_argument(
    "--save_dir", type=str, help="Directory to save plots.",
)
parser.add_argument(
    "--load_plotting_data_dir",
    type=str,
    help="Directory where outputs of evaluate_ensembles.py are saved.",
)
parser.add_argument(
    "--methods", type=str, nargs="+", help="A sequence of method names to plot."
)
parser.add_argument(
    "--dataset", choices=["cifar10", "cifar100", "fmnist", "imagenet", "tiny"], type=str, help="Dataset."
)
parser.add_argument(
    "--runs", type=str, default=[''], nargs='+', help="Subdirectories in load_plotting_data_dir over which to average runs."
)


args = parser.parse_args()

SAVE_DIR = args.save_dir
data_types = ["test"]#, "val"]
ens_attrs = ["evals", "avg_baselearner_evals", "oracle_evals"]
severities = range(6) if (args.dataset in ["cifar10", "cifar100", "tiny"]) else range(1)


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
    "amoebanet_esa": "darkorange",
    "nes_rs_esa": "magenta",
    "darts_esa": "black",
    "darts_rs": "dodgerblue",
    "darts_hyper": "darkorange",
    "joint": "cyan",
    "deepens_darts": "black",
    "deepens_gdas": "cyan",
    "deepens_minimum": "magenta",
    "deepens_amoebanet": "darkorange",
}

markers = {
    'nes_rs': 'v',
    'deepens_rs': 'h',
    'nes_re': 'x',
    "darts_esa": "<",
    "nes_rs_esa": "o",
    'deepens_minimum': '^',
    'deepens_darts': '<',
    'deepens_gdas': '.',
    "darts_rs": "h",
    "darts_hyper": "x",
    "joint": ">",
    'amoebanet_esa': '>',
    'deepens_amoebanet': '>'
}
label_names = {
    'nes_rs': 'NES-RS',
    'deepens_rs': 'DeepEns (RS)',
    "darts_esa": "DeepEns-ESA (DARTS)",
    "amoebanet_esa": "DeepEns-ESA (AmoebaNet)",
    "darts_rs": "NES-RS (depth, width)",
    "joint": "NES + HyperDeepEns",
    "darts_hyper": "HyperDeepEns (lr, wd)",
    "nes_rs_esa": "DeepEns-ESA (RS)",
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

for data_type in data_types:
    for ens_attr in ens_attrs:
        for metric in ["loss", "error", "ece"]:
            #fig, axes = plt.subplots(
            #    len(severities),
            #    len(args.Ms),
            #    figsize=(5.0 * len(args.Ms), 3.5 * len(severities)),
            #    sharex="col",
            #    sharey=False,
            #)

            #if len(severities) == 1:
            #    axes = [axes]

            #if len(args.Ms) == 1:
            #    axes = [[a] for a in axes]

            for i, severity in enumerate(severities):
                for j, M in enumerate(args.Ms):
                    fig, axes = plt.subplots(
                        1,
                        1,
                        figsize=(4.5, 4),
                    )
                    #ax = axes[i][j]
                    ax = axes
                    for pool_name in args.methods:
                        if pool_name in ["nes_rs", "nes_re", "darts_esa",
                                         "amoebanet_esa", "nes_rs_esa",
                                         "darts_rs", "darts_hyper", "joint"]:
                            xs = []
                            ys = []

                            for plot_dir in [os.path.join(args.load_plotting_data_dir, p) for p in args.runs]:

                                if pool_name == 'nes_rs_esa':
                                    if os.path.basename(plot_dir) != "run_3":
                                        continue

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

                                #TODO: remove this when more seeds of darts_esa
                                if pool_name in ["darts_esa", "amoebanet_esa",
                                                "darts_rs", "darts_hyper",
                                                 "joint"]:
                                    break

                            assert len(xs) == len(ys)
                            assert len(set(xs)) == 1

                            x = xs[0]
                            y = np.array(ys).mean(axis=0)
                            err = np.array(ys).std(axis=0)

                            label = f"{label_names[pool_name]}"

                            ax.plot(x, y, label=label, color=colors[pool_name],
                                    linewidth=2, marker=markers[pool_name],
                                    markersize=7, markevery=2)
                            ax.fill_between(x, y - 1.96*err/np.sqrt(len(xs)), y
                                            + 1.96*err/np.sqrt(len(xs)),
                                            color=colors[pool_name], alpha=.15)

                        if pool_name in ["deepens_darts", "deepens_pcdarts",
                                         "deepens_amoebanet", "deepens_gdas",
                                         "deepens_minimum",
                                         #"darts_rs", "darts_hyper", "joint"
                                        ]:
                            if args.runs != ['']:
                                with open(
                                    os.path.join(
                                        args.load_plotting_data_dir,
                                        args.runs[0],
                                        f"plotting_data__M_{M}_pool_{pool_name}.pickle",
                                    ),
                                    "rb",
                                ) as f:
                                    plotting_data = pickle.load(f)
                            else:
                                with open(
                                    os.path.join(
                                        args.load_plotting_data_dir,
                                        f"plotting_data__M_{M}_pool_{pool_name}.pickle",
                                    ),
                                    "rb",
                                ) as f:
                                    plotting_data = pickle.load(f)

                            x = plotting_data[str(M)][str(severity)][ens_attr][pool_name].x
                            yy = plotting_data[str(M)][str(severity)][ens_attr][pool_name].y
                            y = yy[data_type][str(severity)][metric]

                            label = f"{label_names[pool_name]}"
                            if pool_name in ["darts_rs", "darts_hyper"]:
                                label += " no ESA"

                            ax.axhline(
                                y,
                                label=label,
                                linestyle="--",
                                color=colors[pool_name],
                                linewidth=2,
                            )

                        elif pool_name in ["deepens_rs"]:
                            xs = []
                            ys = []

                            for plot_dir in [os.path.join(args.load_plotting_data_dir, p) for p in args.runs]:
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

                            ax.plot(x, mean, label=label, color=colors[pool_name],
                                    linewidth=2, marker=markers[pool_name],
                                    markersize=7, markevery=2)
                            ax.fill_between(x, mean -
                                            1.96*std/np.sqrt(len(xs)), mean
                                            + 1.96*std/np.sqrt(len(xs)),
                                            color=colors[pool_name], alpha=.15)

                    #if i == (len(severities) - 1):
                    ax.set_xlabel("Number of networks evaluated")
                    #if i == 0:
                    #ax.set_title(f"M = {M}")
                    ax.set_title('CIFAR-100')
                    #if j == 0:
                    sev_level = (
                        "(no shift)" if severity == 0 else f"(severity = {severity})"
                    )
                    #ax.set_ylabel(
                    #    "{}".format(ens_attr_to_title[ens_attr])
                    #        #+ "\n"
                    #        + f" {metric_label[metric]}"# {sev_level}"
                    #    )

                    handles, labels = ax.get_legend_handles_labels()
                    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                    ax.legend(handles, labels, framealpha=0.6, fontsize=10)
                    #if len(args.Ms) == 1:
                    #    i = axes[-1]
                    #    plt.setp(i, xlim=(PLOT_EVERY, BUDGET))
                    #    plt.setp(i.xaxis.get_majorticklabels(), ha="right")
                    #else:
                    #    for i in axes[-1]:
                    #        plt.setp(i, xlim=(PLOT_EVERY, BUDGET))
                    #        plt.setp(i.xaxis.get_majorticklabels(), ha="right")
                    #for i in axes[-1]:
                    plt.setp(axes, xlim=(PLOT_EVERY, BUDGET))
                    plt.setp(axes.xaxis.get_majorticklabels(), ha="right")

                    Path(os.path.join(SAVE_DIR, data_type, ens_attr)).mkdir(
                        exist_ok=True, parents=True
                    )
                    plt.tight_layout()
                    fig.savefig(
                        os.path.join(SAVE_DIR, data_type, ens_attr,
                                     f"metric_{metric}_M_{M}_sev_{severity}.pdf"),
                        bbox_inches="tight",
                        pad_inches=0.01,
                    )

                    print("Plot saved.")
                    plt.close("all")

