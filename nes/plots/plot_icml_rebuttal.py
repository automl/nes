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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--esa",
    type=str,
    default="beam_search",
    help="Ensemble selection algorithm. See nes/ensemble_selection/esas.py. Default: beam_search.",
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
data_types = ["val", "test"]
ens_attrs = ["evals", "avg_baselearner_evals", "oracle_evals"]
severities = range(6) if (args.dataset in ["cifar10", "cifar100", "tiny"]) else range(1)

dataset_to_budget = {
    "cifar10": 400,
    "cifar100": 400, 
    "fmnist": 400,
    "tiny": 200,
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
    "deepens_rs": "dodgerblue",
    "nes_re": "crimson",
    "deepens_darts": "black",
    "deepens_darts_anchor": "aqua",
    "deepens_gdas": "cyan",
    "deepens_minimum": "darkgoldenrod",
    "deepens_amoebanet": "darkorange",
    "darts_esa": "black",
    "amoebanet_esa": "darkorange",
    "nes_rs_esa": "dodgerblue",
    "darts_rs": "mediumorchid",
    "darts_hyper": "aqua",
    "joint": "darkorange",
}

markers = {
    'nes_rs': 'v',
    'deepens_rs': 'h',
    'nes_re': 'x',
    'deepens_minimum': '^',
    'deepens_darts': '<',
    'deepens_darts_anchor': '*',
    'deepens_gdas': '.',
    'deepens_amoebanet': '>',
    "darts_esa": "o",
    "amoebanet_esa": "o",
    "nes_rs_esa": "o",
    "darts_rs": "h",
    "darts_hyper": ">",
    "joint": "*",
}
label_names = {
    'nes_rs': 'NES-RS',
    'deepens_rs': 'DeepEns (RS)',
    'nes_re': 'NES-RE',
    'deepens_minimum': 'DeepEns (best arch.)',
    'deepens_darts': 'DeepEns (DARTS)',
    'deepens_darts_anchor': 'AnchoredEns (DARTS)',
    'deepens_gdas': 'DeepEns (GDAS)',
    'deepens_amoebanet': 'DeepEns (AmoebaNet)',
    "darts_esa": "DeepEns+ES (DARTS)",
    "amoebanet_esa": "DeepEns+ES (AmoebaNet)",
    "nes_rs_esa": "DeepEns+ES (RS)",
    "darts_rs": "NES-RS (depth, width)",
    "joint": "NES + HyperEns",
    "darts_hyper": "HyperEns",
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
    'deepens_minimum': 'DeepEns (best arch.)',
    'deepens_darts': 'DeepEns (DARTS)',
    'deepens_darts_anchor': 'AnchoredEns (DARTS)',
    'deepens_gdas': 'DeepEns (GDAS)',
    'deepens_amoebanet': 'DeepEns (AmoebaNet)',
}

if args.dataset == "tiny":
    validation_sizes = [10, 19, 39, 79, 158, 315, 629, 1256, 2506, 5000]
else:
    validation_sizes = [10, 21, 46, 100, 215, 464, 1000, 2154, 4641, 10000]

alphas = list(np.linspace(0.4, 1, len(severities)))
alphas.reverse()

if args.plot_type == "budget":
    alphas = np.linspace(0.2, 1, len(validation_sizes))
    for data_type in data_types:
        for ens_attr in ens_attrs:
            metric = "loss"
            for pool_name in args.methods:
                M = 10
                for i, severity in enumerate(severities):
                    fig, ax = plt.subplots(
                        1,
                        1,
                        figsize=(5., 5.5),
                        sharex="col",
                        sharey=False,
                    )
                    for val_size in validation_sizes:
                        xs = []
                        ys = []
                        runs = args.runs

                        for plot_dir in [os.path.join(args.load_plotting_data_dir, p) for p in runs]:

                            with open(
                                os.path.join(
                                    plot_dir,
                                    f"plotting_data__esa_{args.esa}_M_{M}_pool_{pool_name}_valsize_{val_size}.pickle",
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

                        label = "Val. size = {}".format(val_size)
                        alpha = alphas[list(validation_sizes).index(val_size)]

                        ax.plot(x, y, label=label, color=colors[pool_name],
                                linewidth=2, marker=markers[pool_name],
                                markersize=7, markevery=1 if args.dataset=="tiny" else 2,
                                alpha=alpha)
                        #ax.fill_between(x, y - 1.96*err/np.sqrt(len(xs)), y
                                        #+ 1.96*err/np.sqrt(len(xs)),
                                        #color=colors[pool_name], alpha=0.15)

                    ax.set_xlabel("# nets evaluated")
                    if data_type == "val":
                        title_label = "Validation"
                    else:
                        title_label = "Test"

                    ax.set_title(f"{dataset_to_title[args.dataset]}, "+title_label)
                    sev_level = (
                        "(no shift)" if severity == 0 else f"(severity = {severity})"
                    )
                    ax.set_ylabel(
                        "{}".format(ens_attr_to_title[ens_attr])
                        + f" {metric_label[metric]}", fontsize=17
                    )

                    ax.legend(framealpha=0.6, fontsize=10)

                    plt.setp(ax, xlim=(PLOT_EVERY, BUDGET))
                    plt.setp(ax.xaxis.get_majorticklabels(), ha="right")
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                    Path(os.path.join(SAVE_DIR, data_type, ens_attr)).mkdir(
                        exist_ok=True, parents=True
                    )
                    plt.tight_layout()
                    fig.savefig(
                        os.path.join(SAVE_DIR, data_type, ens_attr,
                                     f"metric_{metric}_sev_{severity}_M_{M}_{pool_name}.pdf"),
                        bbox_inches="tight",
                        pad_inches=0.01,
                    )

                    print("Plot saved.")
                    plt.close("all")

elif args.plot_type == "severity":
    for metric in ["loss", "error"]:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4.5), sharex="col", sharey=False)
        for pool_name in args.methods:
            for i, severity in enumerate(severities):
                y_mean = []
                y_err = []
                for val_size in validation_sizes:
                    xs = []
                    ys = []

                    runs = args.runs

                    for plot_dir in [os.path.join(args.load_plotting_data_dir, p) for p in runs]:

                        with open(
                            os.path.join(
                                plot_dir,
                                f"plotting_data__esa_{args.esa}_M_10_pool_{pool_name}_valsize_{val_size}.pickle",
                            ),
                            "rb",
                        ) as f:
                            plotting_data = pickle.load(f)

                        x = plotting_data["10"][str(severity)]["evals"][args.esa][
                            pool_name
                        ].x
                        yy = plotting_data["10"][str(severity)]["evals"][args.esa][
                            pool_name
                        ].y
                        y = [item["test"][str(severity)][metric] for item in yy]

                        xs.append(x)
                        ys.append(y)

                    assert len(xs) == len(ys)
                    assert len(set(xs)) == 1

                    x = xs[0]
                    y = np.array(ys).mean(axis=0)
                    err = np.array(ys).std(axis=0)

                    y_mean.append(y[-1])
                    y_err.append(1.96*err[-1]/np.sqrt(len(xs)))

                if severity == 0:
                    label = f"{label_names[pool_name]}"
                else:
                    label = None
                #label = "Severity = {}".format(severity)
                color = colors[pool_name]
                marker = markers[pool_name]
                alpha = alphas[severities.index(severity)]

                y_mean = np.array(y_mean)
                y_err = np.array(y_err)

                ax.plot(validation_sizes, y_mean,
                        label=label, 
                        color=color,
                        marker=marker, linewidth=2, markersize=10,
                        markevery=1, ls="-", alpha=alpha)
                ax.fill_between(validation_sizes, y_mean - y_err, y_mean + y_err,
                                color=color, alpha=0.1)

        ax.set_xlabel("Validation size", fontsize=17)
        ax.set_title(f"{dataset_to_title[args.dataset]}")

        sev_level = (
            "(no shift)" if severity == 0 else f"(severity = {severity})"
        )
        ax.set_ylabel(
            "{}".format(ens_attr_to_title["evals"])
            + f" {metric_label[metric]}", fontsize=17
        )

        ax.legend(fontsize=14, framealpha=0.6)

        ax.set_xticks(validation_sizes)
        ax.set_xticklabels(validation_sizes)
        plt.setp(ax, xlim=(10, max(validation_sizes)))
        plt.setp(ax.xaxis.get_majorticklabels(), ha="right")

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax.set_xscale('log')

        Path(os.path.join(SAVE_DIR, "test", "evals")).mkdir(
            exist_ok=True, parents=True
        )
        plt.tight_layout()
        fig.savefig(
            os.path.join(SAVE_DIR, "test", "evals",
                         f"metric_{metric}_M_10.pdf"),
            bbox_inches="tight",
            pad_inches=0.01,
        )

        print("Plot saved.")
        plt.close("all")


#for metric in ["loss", "error"]:
    #for pool_name in args.methods:
        #fig, ax = plt.subplots(1, 1, figsize=(6, 6.5), sharex="col", sharey=False)
        #for val_size in validation_sizes:
            #y_mean = []
            #y_err = []
            #for i, severity in enumerate(severities):
                #xs = []
                #ys = []
#
                #runs = args.runs
#
                #for plot_dir in [os.path.join(args.load_plotting_data_dir, p) for p in runs]:
#
                    #with open(
                        #os.path.join(
                            #plot_dir,
                            #f"plotting_data__esa_{args.esa}_M_10_pool_{pool_name}_valsize_{val_size}.pickle",
                        #),
                        #"rb",
                    #) as f:
                        #plotting_data = pickle.load(f)
#
                    #x = plotting_data["10"][str(severity)]["evals"][args.esa][
                        #pool_name
                    #].x
                    #yy = plotting_data["10"][str(severity)]["evals"][args.esa][
                        #pool_name
                    #].y
                    #y = [item["test"][str(severity)][metric] for item in yy]
#
                    #xs.append(x)
                    #ys.append(y)
#
                #assert len(xs) == len(ys)
                #assert len(set(xs)) == 1
#
                #x = xs[0]
                #y = np.array(ys).mean(axis=0)
                #err = np.array(ys).std(axis=0)
#
                #y_mean.append(y[-1])
                #y_err.append(1.96*err[-1]/np.sqrt(len(xs)))
#
            ##label = f"{label_names[pool_name]}"
            #label = "Val. size = {}".format(val_size)
            #color = colors[pool_name]
            #marker = markers[pool_name]
            #alpha = alphas[list(validation_sizes).index(val_size)]
#
            #y_mean = np.array(y_mean)
            #y_err = np.array(y_err)
            #print(y_mean)
#
            #ls = "-"
#
            #ax.plot(severities, y_mean, label=label, color=color,
                    #marker=marker, linewidth=2, markersize=10,
                    #markevery=1, ls=ls, alpha=alpha)
            ##ax.fill_between(severities, y_mean - y_err, y_mean + y_err,
                            ##color=color, alpha=0.15)
#
            #ax.set_xlabel("Shift severity", fontsize=20)
            #ax.set_title(f"{dataset_to_title[args.dataset]}, {label_names[pool_name]}")
#
            #ax.set_ylabel(
                #"{}".format(ens_attr_to_title["evals"])
                #+ f" {metric_label[metric]}"
            #)
            #sev_level = (
                #"(no shift)" if severity == 0 else f"(severity = {severity})"
            #)
            #ax.set_ylabel(
                #"{}".format(ens_attr_to_title["evals"])
                #+ f" {metric_label[metric]}", fontsize=17
            #)
#
        #ax.legend(fontsize=12)
#
        #ax.set_xticks(severities)
        #ax.set_xticklabels(severities)
        #plt.setp(ax, xlim=(0, max(severities)))
        #plt.setp(ax.xaxis.get_majorticklabels(), ha="right")
#
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#
        #Path(os.path.join(SAVE_DIR, "test", "evals")).mkdir(
            #exist_ok=True, parents=True
        #)
        #plt.tight_layout()
        #fig.savefig(
            #os.path.join(SAVE_DIR, "test", "evals",
                         #f"metric_{metric}_M_10_{pool_name}.pdf"),
            #bbox_inches="tight",
            #pad_inches=0.01,
        #)
#
        #print("Plot saved.")
        #plt.close("all")
