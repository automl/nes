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
    "--esa", nargs="+", type=str,
    help="ESA."
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
severities = range(6) if (args.dataset in ["cifar10", "cifar100", "tiny"]) else range(1)

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
    "nes_rs_200": "forestgreen",
    "deepens_rs": "dodgerblue",
    "nes_re": "crimson",
    "nes_re_200": "crimson",
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
    'nes_rs_200': 'v',
    'deepens_rs': 'h',
    'nes_re': 'x',
    'nes_re_200': 'x',
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
    "beam_search": "x",
    "beam_search_with_div_0.05": "P",
    "beam_search_with_div_0.1": "o",
    "beam_search_with_div_0.5": "s",
    "beam_search_with_div_0.75": "h",
    "beam_search_with_div_1.0": ">",
    "beam_search_with_div_1.5": ".",
    "beam_search_with_div_2.0": "d",
    "beam_search_with_div_5.0": "v",
    "beam_search_with_div_10.0": "*",
    "beam_search_with_div_50.0": "^",
    "beam_search_bma_loss": ">",
    "beam_search_bma_acc": "<",
    "quick_and_greedy": "o",
    "forw_select_replace": "*",
    "top_M": "v",
    "linear_unweighted_stack": "h",
    "linear_weighted_stack": "^",
}
label_names = {
    'nes_rs': 'NES-RS (K = 400)',
    'nes_rs_200': 'NES-RS (K = 200)',
    'deepens_rs': 'DeepEns (RS)',
    'nes_re': 'NES-RE (K = 400)',
    'nes_re_200': 'NES-RE (K = 200)',
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
    ('nes_rs', 'beam_search'): 'NES-RS (ForwardSelect)',
    ('nes_rs', 'beam_search_with_div_0.05'): 'NES-RS (ForwardSelect w/ div 0.05)',
    ('nes_rs', 'beam_search_with_div_0.1'): 'NES-RS (ForwardSelect w/ div 0.1)',
    ('nes_rs', 'beam_search_with_div_0.5'): 'NES-RS (ForwardSelect w/ div 0.5)',
    ('nes_rs', 'beam_search_with_div_0.75'): 'NES-RS (ForwardSelect w/ div 0.75)',
    ('nes_rs', 'beam_search_with_div_1.0'): 'NES-RS (ForwardSelect w/ div 1.0)',
    ('nes_rs', 'beam_search_with_div_1.5'): 'NES-RS (ForwardSelect w/ div 1.5)',
    ('nes_rs', 'beam_search_with_div_2.0'): 'NES-RS (ForwardSelect w/ div 2.0)',
    ('nes_rs', 'beam_search_with_div_5.0'): 'NES-RS (ForwardSelect w/ div 5.0)',
    ('nes_rs', 'beam_search_with_div_10.0'): 'NES-RS (ForwardSelect w/ div 10.0)',
    ('nes_rs', 'beam_search_with_div_50.0'): 'NES-RS (ForwardSelect w/ div 50.0)',
    ('nes_rs', 'beam_search_bma_acc'): 'NES-RS (BMA accuracy)',
    ('nes_rs', 'beam_search_bma_loss'): 'NES-RS (BMA Loss)',
    ('nes_rs', 'quick_and_greedy'): 'NES-RS (Quick and Greedy)',
    ('nes_rs', 'top_M'): 'NES-RS (Top M)',
    ('nes_rs', 'forw_select_replace'): 'NES-RS (Forward w/ repl.)',
    ('nes_rs', 'linear_unweighted_stack'): 'NES-RS (Unweighted Stack)',
    ('nes_rs', 'linear_weighted_stack'): 'NES-RS (Weighted Stack)',
    ('nes_re', 'beam_search'): 'NES-RE (ForwardSelect)',
    ('nes_re', 'beam_search_with_div_0.05'): 'NES-RE (ForwardSelect w/ div 0.05)',
    ('nes_re', 'beam_search_with_div_0.1'): 'NES-RE (ForwardSelect w/ div 0.1)',
    ('nes_re', 'beam_search_with_div_0.5'): 'NES-RE (ForwardSelect w/ div 0.5)',
    ('nes_re', 'beam_search_with_div_0.75'): 'NES-RE (ForwardSelect w/ div 0.75)',
    ('nes_re', 'beam_search_with_div_1.0'): 'NES-RE (ForwardSelect w/ div 1.0)',
    ('nes_re', 'beam_search_with_div_1.5'): 'NES-RE (ForwardSelect w/ div 1.5)',
    ('nes_re', 'beam_search_with_div_2.0'): 'NES-RE (ForwardSelect w/ div 2.0)',
    ('nes_re', 'beam_search_with_div_5.0'): 'NES-RE (ForwardSelect w/ div 5.0)',
    ('nes_re', 'beam_search_with_div_10.0'): 'NES-RE (ForwardSelect w/ div 10.0)',
    ('nes_re', 'beam_search_with_div_50.0'): 'NES-RE (ForwardSelect w/ div 50.0)',
    ('nes_re', 'beam_search_bma_acc'): 'NES-RE (BMA accuracy)',
    ('nes_re', 'beam_search_bma_loss'): 'NES-RE (BMA Loss)',
    ('nes_re', 'quick_and_greedy'): 'NES-RE (Quick and Greedy)',
    ('nes_re', 'top_M'): 'NES-RE (Top M)',
    ('nes_re', 'forw_select_replace'): 'NES-RE (Forward w/ repl.)',
    ('nes_re', 'linear_unweighted_stack'): 'NES-RE (Unweighted Stack)',
    ('nes_re', 'linear_weighted_stack'): 'NES-RE (Weighted Stack)',
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
    "beam_search_with_div_1.0": "Forward Select with div 1.0",
    "beam_search_with_div_1.5": "Forward Select with div 1.5",
    "beam_search_with_div_5.0": "Forward Select with div 5.0",
    "beam_search_bma_loss": "BMA Loss",
    "beam_search_bma_acc": "BMA Accuracy",
    "top_M": "Top M",
    "forw_select_replace": "Forward Select (w/ repl.)",
    "quick_and_greedy": "Quick and Greedy",
    "linear_unweighted_stack": "Linear Unweighted Stacking",
    "linear_weighted_stack": "Linear Weighted Stacking",
}

esa_to_linestyle = {
    "beam_search": "-",
    "beam_search_with_div_0.05": ":",
    "beam_search_with_div_0.1": ":",
    "beam_search_with_div_0.5": ":",
    "beam_search_with_div_0.75": ":",
    "beam_search_with_div_1.0": ":",
    "beam_search_with_div_1.5": "-.",
    "beam_search_with_div_2.0": ":",
    "beam_search_with_div_5.0": "--",
    "beam_search_with_div_10.0": "--",
    "beam_search_with_div_50.0": "--",
    "beam_search_bma_acc": ":",
    "beam_search_bma_loss": "-.",
    "top_M": "--",
    "quick_and_greedy": "dotted",
    "linear_unweighted_stack": "-.",
    "linear_weighted_stack": "-",
    "forw_select_replace": ":",
}

esa_list = [
    "beam_search",
    #"beam_search_with_div_0.05",
    "beam_search_with_div_0.1",
    "beam_search_with_div_0.5",
    #"beam_search_with_div_0.75",
    "beam_search_with_div_1.0",
    "beam_search_with_div_1.5",
    "beam_search_with_div_2.0",
    "beam_search_with_div_5.0",
    "beam_search_with_div_10.0",
    #"beam_search_with_div_50.0"
]

color_list = list(plt.cm.rainbow(np.linspace(0, 1, len(esa_list))))

esa_to_alpha = list(np.linspace(0, 0.6, len(esa_list)))

linestyle_method = {
    'deepens_minimum': 'DeepEns (best arch.)',
    'deepens_darts': 'DeepEns (DARTS)',
    'deepens_darts_anchor': 'AnchoredEns (DARTS)',
    'deepens_gdas': 'DeepEns (GDAS)',
    'deepens_amoebanet': 'DeepEns (AmoebaNet)',
}



for data_type in data_types:
    for ens_attr in ens_attrs:
        for metric in ["loss", "error", "ece", "normalized_disagreement"]:
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

                for esa in args.esa:

                    for pool_name in args.methods:
                        y_mean = []
                        y_err = []

                        for j, M in enumerate(args.Ms):

                            if pool_name in ["nes_re", "nes_rs", "nes_re_200",
                                             "nes_rs_200",
                                             "amoebanet_esa", "darts_esa",
                                             "nes_rs_esa",
                                             "darts_rs", "darts_hyper", "joint"]:
                                xs = []
                                ys = []

                                if pool_name in ["nes_re", "nes_rs"]:
                                    runs = args.runs
                                elif pool_name == "nes_rs_esa":
                                    runs = ["run_3"]
                                elif pool_name in ["nes_re_200", "nes_rs_200"]:
                                    runs = args.runs
                                else:
                                    runs = ["run_1"]

                                for plot_dir in [os.path.join(args.load_plotting_data_dir, p) for p in runs]:
                                    with open(
                                        os.path.join(
                                            plot_dir,
                                            f"plotting_data__esa_{esa}_M_{M}_pool_{pool_name}.pickle",
                                        ),
                                        "rb",
                                    ) as f:
                                        plotting_data = pickle.load(f)

                                    pool_name_tmp = pool_name.strip('_200')

                                    x = plotting_data[str(M)][str(severity)][ens_attr][esa][
                                        pool_name_tmp
                                    ].x
                                    yy = plotting_data[str(M)][str(severity)][ens_attr][esa][
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

                                #label = f"{label_names[pool_name]}"
                                label = label_names[(pool_name, esa)]

                                y_mean.append(y[-1])
                                y_err.append(1.96*err[-1]/np.sqrt(len(xs)))


                            elif pool_name in ["deepens_darts", "deepens_pcdarts",
                                                "deepens_amoebanet", "deepens_gdas",
                                                "deepens_darts_anchor", "deepens_minimum"]:
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
                                if pool_name == 'deepens_darts_anchor' and \
                                metric == 'error' and M == 10 and ens_attr == 'evals':
                                    print(severity)
                                    print(y)


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

                                    x = x + [BUDGET]
                                    y = y + [y[-1]]

                                    xs.append(x)
                                    ys.append(y)

                                x, all_pools = get_trajectories(ys, xs)

                                std = all_pools.std(axis=0)
                                mean = np.mean(all_pools, axis=0)

                                #label = f"{label_names[pool_name]}"
                                label = label_names[(pool_name, esa)]


                                y_mean.append(mean[-1])
                                y_err.append(1.96*std[-1]/np.sqrt(len(xs)))

                        #color = colors[pool_name]
                        color = color_list[esa_list.index(esa)]
                        marker = markers[esa]

                        y_mean = np.array(y_mean)
                        y_err = np.array(y_err)

                        if "esa" in pool_name:
                            ls = "dotted"
                        elif "_200" in pool_name:
                            ls = "--"
                        else:
                            ls = "-"

                        ls = esa_to_linestyle[esa]

                        if '_200' in pool_name:
                            alpha_1 = 0.55
                            alpha_2 = 0.0
                        else:
                            alpha_1 = 1.0
                            alpha_2 = 0.15

                        alpha = 1. - esa_to_alpha[esa_list.index(esa)]

                        ax.plot(args.Ms, y_mean, label=label, color=color,
                                marker=marker, linewidth=2, markersize=7,
                                markevery=1,
                                #ls=ls, 
                                alpha=alpha)
                        #ax.fill_between(args.Ms, y_mean - y_err, y_mean +
                                        #y_err, color=color, alpha=alpha_2)

                        ax.set_xlabel("Ensemble size (M)", fontsize=17)

                        #ax.set_title(
                            #dataset_to_title[args.dataset] + ', %s'%(esa_to_title[args.esa])
                        #)
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

                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                #labels_new = tuple(['DeepEns (RS)', 'DeepEns (AmoebaNet)',
                #                    'DeepEns (DARTS)',
                #                    'NES-RS (K = 400)', 'NES-RE (K = 400)'])
                if args.methods == "nes_re":
                    labels_new = tuple(['NES-RE (ForwardSelect)'] + ['NES-RE (ForwardSelect w/ div {})'.format(x) for
                                       x in [0.1, 0.5, 1.0, 1.5,
                                             2.0, 5.0, 10.0]])
                else:
                    labels_new = tuple(['NES-RS (ForwardSelect)'] + ['NES-RS (ForwardSelect w/ div {})'.format(x) for
                                       x in [0.1, 0.5, 1.0, 1.5,
                                             2.0, 5.0, 10.0]])
                idx_old = [labels.index(x) for x in labels_new]
                labels = labels_new
                handles = tuple([handles[i] for i in idx_old])

                if (metric == 'loss') and (ens_attr == "evals") and (args.dataset == "cifar10"):
                    ax.legend(handles, labels, framealpha=0.3, fontsize=12)

                if (metric == 'loss') and (ens_attr == 'evals') and \
                   (args.dataset == 'tiny') and ('darts_esa' in args.methods):
                    #handles, labels = ax.get_legend_handles_labels()
                    #labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                    #labels_new = tuple(['DeepEns (DARTS)', 'DeepEns+ES (DARTS)',
                                        #'DeepEns (AmoebaNet)',
                                        #'DeepEns+ES (AmoebaNet)', 'DeepEns (RS)',
                                        #'DeepEns+ES (RS)',
                                        #'NES-RS (K = 400)',
                                        #'NES-RE (K = 400)'])
                    #idx_old = [labels.index(x) for x in labels_new]
                    #labels = labels_new
                    #handles = tuple([handles[i] for i in idx_old])
                    ax.legend(handles, labels, framealpha=0.45, fontsize=11, ncol=2, handletextpad=0.1, columnspacing=0.2,
                              loc='best')
                ax.legend(handles, labels, framealpha=0.45, fontsize=10, ncol=1, handletextpad=0.1, columnspacing=0.2,
                            loc='upper right')

                plt.setp(ax, xlim=(min(args.Ms), max(args.Ms)))
                plt.setp(ax.xaxis.get_majorticklabels(), ha="right")

                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                Path(os.path.join(SAVE_DIR, data_type, ens_attr)).mkdir(
                    exist_ok=True, parents=True
                )
                plt.tight_layout()
                fig.savefig(
                    os.path.join(SAVE_DIR, data_type, ens_attr, f"metric_{metric}_sev_{severity}.pdf"),
                    bbox_inches="tight",
                    pad_inches=0.01,
                )

                print("Plot saved.")
                plt.close("all")

