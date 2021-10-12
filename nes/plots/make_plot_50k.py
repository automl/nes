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
plt.rcParams['font.size'] = 15
#matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
#matplotlib.rc("text", usetex=True)


parser = argparse.ArgumentParser()
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
parser.add_argument(
    "--plot_type", type=str, choices=["budget", "ensemble_size", "severity"], help="Which type of plots to make."
)


args = parser.parse_args()

SAVE_DIR = args.save_dir
#data_types = ["val", "test"]
data_types = ["test"]
#ens_attrs = ["evals", "avg_baselearner_evals", "oracle_evals", "disagreement"]
ens_attrs = ["evals"]
#severities = range(6) if (args.dataset in ["cifar10", "cifar100", "tiny"]) else range(1)
severities = range(1)

args.esa = "beam_search"

dataset_to_budget = {
    "cifar10": 400,
    "cifar100": 400,
    "fmnist": 400,
    "tiny": 400,
    "imagenet": 1000
}

BUDGET = dataset_to_budget[args.dataset]
PLOT_EVERY = 25

plot_individual_lines = False

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
plt.rcParams["font.size"] = 15


metric_label = {"loss": "NLL", "error": "Error", "ece": "ECE"}

colors = {
    "nes_rs": "forestgreen",
    "nes_re_50k": "crimson",
    "nes_rs_50k": "forestgreen",
    "nes_rs_200": "forestgreen",
    "deepens_rs": "dodgerblue",
    "nes_re": "crimson",
    "nes_re_200": "crimson",
    "deepens_darts": "black",
    "deepens_darts_50k": "black",
    "deepens_darts_anchor": "aqua",
    "deepens_gdas": "cyan",
    "deepens_minimum": "darkgoldenrod",
    "deepens_amoebanet": "darkorange",
    "deepens_amoebanet_50k": "darkorange",
    "darts_esa": "black",
    "amoebanet_esa": "darkorange",
    "nes_rs_esa": "dodgerblue",
    "darts_rs": "mediumorchid",
    "darts_hyper": "aqua",
    "joint": "darkorange",
}

markers = {
    'nes_rs': 'v',
    'nes_rs_50k': 'o',
    'nes_re_50k': '>',
    'nes_rs_200': 'v',
    'deepens_rs': 'h',
    'nes_re': 'h',
    'nes_re_200': 'x',
    'deepens_minimum': '^',
    'deepens_darts': '<',
    'deepens_darts_50k': 'x',
    'deepens_darts_anchor': '*',
    'deepens_gdas': '.',
    'deepens_amoebanet': '>',
    'deepens_amoebanet_50k': 'h',
    "darts_esa": "o",
    "amoebanet_esa": "o",
    "nes_rs_esa": "o",
    "darts_rs": "h",
    "darts_hyper": ">",
    "joint": "*",
    "beam_search": "x",
    "quick_and_greedy": "o",
    "top_M": "v",
}
label_names = {
    'nes_rs': r'NES-RS ($\mathcal{D}_{train}$)',
    'nes_rs_50k': r"NES-RS ($\mathcal{D}_{train}+\mathcal{D}_{val}$)",
    'nes_re_50k': r"NES-RE ($\mathcal{D}_{train}+\mathcal{D}_{val}$)",
    'nes_rs_200': 'NES-RS (K = 200)',
    'deepens_rs': 'DeepEns (RS)',
    'nes_re': r'NES-RE ($\mathcal{D}_{train}$)',
    'nes_re_200': 'NES-RE (K = 200)',
    'deepens_minimum': 'DeepEns (best arch.)',
    'deepens_darts': r"DeepEns (DARTS) ($\mathcal{D}_{train}$)",
    'deepens_darts_50k': r'DeepEns (DARTS) $(\mathcal{D}_{train}+\mathcal{D}_{val})$',
    'deepens_darts_anchor': 'AnchoredEns (DARTS)',
    'deepens_gdas': 'DeepEns (GDAS)',
    'deepens_amoebanet': 'DeepEns (AmoebaNet) (40k)',
    'deepens_amoebanet_50k': 'DeepEns (AmoebaNet) (50k)',
    "darts_esa": "DeepEns+ES (DARTS)",
    "amoebanet_esa": "DeepEns+ES (AmoebaNet)",
    "nes_rs_esa": "DeepEns+ES (RS)",
    "darts_rs": "NES-RS (depth, width)",
    "joint": "NES + HyperEns",
    "darts_hyper": "HyperEns",
    ('nes_rs', 'beam_search'): 'NES-RS (Forward w/o repl.)',
    ('nes_rs', 'quick_and_greedy'): 'NES-RS (Quick and Greedy)',
    ('nes_rs', 'top_M'): 'NES-RS (Top M)',
    ('nes_re', 'beam_search'): 'NES-RE (Forward w/o repl.)',
    ('nes_re', 'quick_and_greedy'): 'NES-RE (Quick and Greedy)',
    ('nes_re', 'top_M'): 'NES-RE (Top M)',
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

esa_to_title = {
    "beam_search": "Forward Select (w/o repl.)",
    "top_M": "Top M",
    "quick_and_greedy": "Quick and Greedy",
}

esa_to_linestyle = {
    "beam_search": "-",
    "top_M": "--",
    "quick_and_greedy": "dotted",
}

linestyle_method = {
    'deepens_minimum': 'DeepEns (best arch.)',
    'deepens_darts': 'DeepEns (DARTS)',
    'deepens_darts_anchor': 'AnchoredEns (DARTS)',
    'deepens_gdas': 'DeepEns (GDAS)',
    'deepens_amoebanet': 'DeepEns (AmoebaNet)',
}



for data_type in data_types:
    for ens_attr in ens_attrs:
        for metric in ["loss", "error", "ece"]:
            if (ens_attr == "disagreement") ^ ("disagreement" in metric):
                continue

            for i, severity in enumerate(severities):
                fig, ax = plt.subplots(
                1,
                1,
                figsize=(6, 6.5),
                sharex="col",
                sharey=False,
                )

                for pool_name in args.methods:
                    y_mean = []
                    y_err = []

                    for j, M in enumerate(args.Ms):
                        if pool_name in ["nes_re", "nes_rs", "nes_re_200",
                                         "nes_rs_200", "nes_re_50k",
                                         "amoebanet_esa", "darts_esa",
                                         "nes_rs_esa", "nes_rs_50k",
                                         "darts_rs", "darts_hyper", "joint"]:
                            xs = []
                            ys = []

                            for plot_dir in [os.path.join(args.load_plotting_data_dir, p) for p in args.runs]:
                                with open(
                                    os.path.join(
                                        plot_dir,
                                        f"plotting_data__esa_{args.esa}_M_{M}_pool_{pool_name}.pickle",
                                    ),
                                    "rb",
                                ) as f:
                                    plotting_data = pickle.load(f)

                                pool_name_tmp = pool_name.strip('_200')

                                x = plotting_data[str(M)][str(severity)][ens_attr][args.esa][
                                    pool_name_tmp
                                ].x
                                yy = plotting_data[str(M)][str(severity)][ens_attr][args.esa][
                                    pool_name_tmp
                                ].y
                                y = [item[data_type][str(severity)][metric] for item in yy]

                                xs.append(x)
                                ys.append(y)

                            assert len(xs) == len(ys)
                            assert len(set(xs)) == 1

                            x = xs[0]
                            y = np.array(ys).mean(axis=0)
                            err = np.array(ys).std(axis=0)

                            label = f"{label_names[pool_name]}"
                            y_mean.append(y[-1])
                            y_err.append(1.96*err[-1]/np.sqrt(len(xs)))

                        else:
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

                            y_mean.append(y)
                            y_err.append(0) # don't have error bars for these

                    color = colors[pool_name]
                    marker = markers[pool_name]

                    y_mean = np.array(y_mean)
                    y_err = np.array(y_err)

                    if "50k" in pool_name:
                        ls = "dotted"
                    else:
                        ls = "-"

                    alpha_1 = 1.0
                    alpha_2 = 0.15


                    ax.plot(args.Ms, y_mean, label=label, color=color,
                            marker=marker, linewidth=2, markersize=7,
                            markevery=1, ls=ls, alpha=alpha_1)
                    ax.fill_between(args.Ms, y_mean - y_err, y_mean +
                                    y_err, color=color, alpha=alpha_2)

                    ax.set_xlabel("Ensemble size (M)", fontsize=17)

                    ax.set_title(
                        dataset_to_title[args.dataset]
                    )

                    if (ens_attr in ["oracle_evals",
                                     "avg_baselearner_evals", "evals"]):
                        sev_level = (
                            "(no shift)" if severity == 0 else f"(severity = {severity})"
                        )
                        ax.set_ylabel(
                            "{}".format(ens_attr_to_title[ens_attr])
                            + f" {metric_label[metric]}", fontsize=17
                        )
                    elif ens_attr == "disagreement":
                        ax.set_ylabel(
                            "Pred. disagreement", fontsize=17
                        )

                handles, labels = ax.get_legend_handles_labels()
                #labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                #labels_new = tuple(['DeepEns (DARTS) (40k)', 'DeepEns (DARTS) (50k)', 'NES-RS (40k)', 'NES-RS (50k)'])
                #idx_old = [labels.index(x) for x in labels_new]
                #labels = labels_new
                #handles = tuple([handles[i] for i in idx_old])

                ax.legend(handles, labels, framealpha=0.45, fontsize=12,
                          ncol=1, handletextpad=0.1, columnspacing=0.2,
                          loc='upper right')

                plt.setp(ax, xlim=(min(args.Ms), max(args.Ms)))
                plt.setp(ax.xaxis.get_majorticklabels(), ha="right")

                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                Path(os.path.join(SAVE_DIR, data_type, ens_attr)).mkdir(
                    exist_ok=True, parents=True
                )
                plt.tight_layout()
                fig.savefig(
                    os.path.join(SAVE_DIR, data_type, ens_attr,
                                 f"metric_{metric}_sev_{severity}.pdf"),
                    bbox_inches="tight",
                    pad_inches=0.01,
                )

                print("Plot saved.")
                plt.close("all")

