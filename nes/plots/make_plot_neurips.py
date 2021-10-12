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
ens_attrs = ["evals", "avg_baselearner_evals", "oracle_evals", "disagreement"]
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

# # ===================================
# # Get C10 old data
# # ===================================

# if args.dataset == "cifar10":
    


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


if args.plot_type == "budget":
    for data_type in data_types:
        for ens_attr in ens_attrs:
            for metric in ["loss", "error", "ece", "normalized_disagreement"]:
                if (ens_attr == "disagreement") ^ ("disagreement" in metric):
                    continue
                # fig, axes = plt.subplots(
                #     len(severities),
                #     len(args.Ms),
                #     figsize=(5.0 * len(args.Ms), 3.5 * len(severities)),
                #     sharex="col",
                #     sharey=False,
                # )

                # if len(severities) == 1:
                #     axes = [axes]

                # if len(args.Ms) == 1:
                #     axes = [[a] for a in axes]

                for i, severity in enumerate(severities):
                    for j, M in enumerate(args.Ms):
                        fig, ax = plt.subplots(
                            1,
                            1,
                            figsize=(4., 4.5),
                            sharex="col",
                            sharey=False,
                        )

                        # if not (args.dataset == "tiny" and severity == 5 and data_type == "test" and ens_attr == "evals" and metric == "loss" and M == 10):
                        #     continue

                        # ax = axes[i][j]
                        for pool_name in args.methods:
                            if pool_name in ["nes_re", "nes_rs",
                                             "amoebanet_esa", "darts_esa",
                                             "nes_rs_esa",
                                             "darts_rs", "darts_hyper", "joint"]:
                                xs = []
                                ys = []


                                if pool_name in ["nes_re", "nes_rs"]:
                                    runs = args.runs
                                elif pool_name == "nes_rs_esa":
                                    runs = ["run_3"]
                                else:
                                    runs = ["run_1"]

                                for plot_dir in [os.path.join(args.load_plotting_data_dir, p) for p in runs]:

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
                                        markersize=7, markevery=1 if args.dataset=="tiny" else 2)
                                ax.fill_between(x, y - 1.96*err/np.sqrt(len(xs)), y
                                                + 1.96*err/np.sqrt(len(xs)),
                                                color=colors[pool_name], alpha=0.15)

                            elif pool_name in ["deepens_darts", "deepens_pcdarts",
                                               "deepens_amoebanet", "deepens_gdas",
                                               "deepens_minimum", "deepens_darts_anchor"]:
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

                                if pool_name == 'deepens_darts_anchor':
                                    ls = ':'
                                elif pool_name == 'deepens_darts':
                                    ls = "--"
                                else:
                                    ls = "-."


                                ax.axhline(
                                    y,
                                    label=label,
                                    linestyle=ls,
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

                                x_1, all_pools = get_trajectories(ys, xs)

                                std = all_pools.std(axis=0)
                                mean = np.mean(all_pools, axis=0)

                                #########################################
                                xs = []
                                ys = []

                                pool_name = "nes_rs_esa"

                                with open(
                                    os.path.join(
                                        args.load_plotting_data_dir,
                                        "run_3",
                                        f"plotting_data__esa_{args.esa}_M_{M}_pool_nes_rs_esa.pickle",
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

                                x = list(x_1[:-1]) + [i + 200 for i in list(xs[0])]
                                y = list(mean[:-1]) + list(np.array(ys).mean(axis=0))
                                err = list(std[:-1]) + list(np.array(ys).std(axis=0))

                                label = f"{label_names[pool_name]}"

                                ax.plot(x, y, label=label, color=colors[pool_name],
                                        linewidth=2, marker=markers[pool_name],
                                        markersize=7, markevery=1 if args.dataset=="tiny" else 2)
                                #ax.fill_between(x, y - 1.96*err/np.sqrt(len(xs)), y
                                #                + 1.96*err/np.sqrt(len(xs)),
                                #                color=colors[pool_name], alpha=0.15)
                                ######################################################

                        # if i == (len(severities) - 1):
                        ax.set_xlabel("# nets evaluated")
                        if data_type == "val":
                            title_label = "Validation"
                        else:
                            title_label = "Test"

                        ax.set_title(f"{dataset_to_title[args.dataset]}, "+title_label)
                        if (ens_attr in ["oracle_evals",
                                         "avg_baselearner_evals", "evals"]): # and (args.dataset == 'cifar10'):
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


                        # # if severity == 5 and args.dataset != "cifar100":
                        # handles, labels = ax.get_legend_handles_labels()
                        # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                        # # if args.dataset == "tiny" and severity == 5 and data_type == "test" and ens_attr == "evals" and metric == "loss" and M == 10:
                        # ax.legend(handles, labels, framealpha=0.6, fontsize=10, loc="upper right")
                        #     # else:
                        #     #     ax.legend(handles, labels, framealpha=0.6, fontsize=10)



                        handles, labels = ax.get_legend_handles_labels()
                        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                        #if 'deepens_minimum' not in args.methods:
                        #labels_new = tuple(['DeepEns (RS)', 'DeepEns (AmoebaNet)',
                                                #'DeepEns (DARTS)', 'AnchoredEns (DARTS)',
                                                #'NES-RS', 'NES-RE'])
                        #else:
                            #labels_new = tuple(['DeepEns (RS)', 'DeepEns (GDAS)',
                                                #'DeepEns (best arch.)',
                                                #'NES-RS', 'NES-RE'])
                        #idx_old = [labels.index(x) for x in labels_new]
                        #labels = labels_new
                        #handles = tuple([handles[i] for i in idx_old])

                        if args.dataset == "cifar10" and ens_attr == "evals":
                            if severity == 0:
                                ax.legend(handles, labels, framealpha=0.6, fontsize=10)
                        if args.dataset == "cifar100" and metric == "loss" and ens_attr == "evals":
                            if severity == 0:
                                ax.legend(handles, labels, framealpha=0.6, fontsize=10)
                        #if len(args.Ms) == 1:
                        #    i = axes[-1]
                        #    plt.setp(i, xlim=(PLOT_EVERY, BUDGET))
                        #    plt.setp(i.xaxis.get_majorticklabels(), ha="right")
                        #else:
                        #    for i in axes[-1]:
                        #        plt.setp(i, xlim=(PLOT_EVERY, BUDGET))
                        #        plt.setp(i.xaxis.get_majorticklabels(), ha="right")
                        # for i in axes[-1]:
                        if args.dataset == "tiny" and severity == 5 and data_type == "test" and ens_attr == "evals" and metric == "loss" and M == 10:
                            #ax.legend(handles, labels, framealpha=0.6, fontsize=10, loc="upper right")
                            plt.setp(ax, ylim=(3.51, 3.80)) # fix ylim in /tiny/budget/test/evals/metric_loss_sev_5_M_10
                        if args.dataset == "tiny" and severity == 0 and data_type == "test" and ens_attr == "evals" and metric == "loss" and M == 10:
                            ax.legend(handles, labels, framealpha=0.6, fontsize=10)

                        plt.setp(ax, xlim=(PLOT_EVERY, BUDGET))
                        plt.setp(ax.xaxis.get_majorticklabels(), ha="right")
                        
                        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


                        Path(os.path.join(SAVE_DIR, data_type, ens_attr)).mkdir(
                            exist_ok=True, parents=True
                        )
                        plt.tight_layout()
                        fig.savefig(
                            os.path.join(SAVE_DIR, data_type, ens_attr, f"metric_{metric}_sev_{severity}_M_{M}.pdf"),
                            bbox_inches="tight",
                            pad_inches=0.01,
                        )

                        print("Plot saved.")
                        plt.close("all")

elif args.plot_type == "ensemble_size":
    for data_type in data_types:
        for ens_attr in ens_attrs:
            for metric in ["loss", "error", "ece", "normalized_disagreement"]:
                if (ens_attr == "disagreement") ^ ("disagreement" in metric):
                    continue
                # fig, axes = plt.subplots(
                #     len(severities),
                #     1,
                #     figsize=(5.0, 3.5 * len(severities)),
                #     sharex="col",
                #     sharey=False,
                # )

                for i, severity in enumerate(severities):
                    fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(6, 6.5),# * len(severities)), (4, 4.5)
                    sharex="col",
                    sharey=False,
                    )

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

                                # ax.axhline(
                                #     y,
                                #     label=label,
                                #     linestyle="--",
                                #     color=colors[pool_name],
                                #     linewidth=2,
                                # )

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

                        color = colors[pool_name]
                        marker = markers[pool_name]

                        y_mean = np.array(y_mean)
                        y_err = np.array(y_err)

                        if "esa" in pool_name:
                            ls = "dotted"
                        elif "_200" in pool_name:
                            ls = "--"
                        else:
                            ls = "-"

                        if '_200' in pool_name:
                            alpha_1 = 0.55
                            alpha_2 = 0.0
                        else:
                            alpha_1 = 1.0
                            alpha_2 = 0.15


                        ax.plot(args.Ms, y_mean, label=label, color=color,
                                marker=marker, linewidth=2, markersize=7,
                                markevery=1, ls=ls, alpha=alpha_1)
                        ax.fill_between(args.Ms, y_mean - y_err, y_mean +
                                        y_err, color=color, alpha=alpha_2)

                        # if i == (len(severities) - 1):
                        ax.set_xlabel("Ensemble size (M)", fontsize=17)

                        ax.set_title(dataset_to_title[args.dataset])
                        # if i == 0:
                        #     ax.set_title(f"M = {M}")

                        if (ens_attr in ["oracle_evals",
                                         "avg_baselearner_evals", "evals"]): # and (args.dataset == 'cifar10'):
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

                    #if (metric == 'loss') and (ens_attr == "oracle_evals") and (args.dataset == "tiny"):
                    #    handles, labels = ax.get_legend_handles_labels()
                    #    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                    #    ax.legend(handles, labels, framealpha=0.6, fontsize=10)
                    handles, labels = ax.get_legend_handles_labels()
                    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                    if 'deepens_darts_anchor' in args.methods:
                        labels_new = tuple(['DeepEns (RS)', 'DeepEns (AmoebaNet)',
                                            'DeepEns (DARTS)', 'AnchoredEns (DARTS)',
                                            'NES-RS (K = 200)', 'NES-RE (K = 200)'])
                    else:
                        labels_new = tuple(['DeepEns (RS)', 'DeepEns (AmoebaNet)',
                                            'DeepEns (DARTS)', 'NES-RS (K = 200)',
                                            'NES-RS (K = 400)', 'NES-RE (K = 200)', 'NES-RE (K = 400)'])
                    idx_old = [labels.index(x) for x in labels_new]
                    labels = labels_new
                    handles = tuple([handles[i] for i in idx_old])
                    if (metric == 'loss') and (ens_attr == "evals") and (args.dataset == "cifar10"):
                        ax.legend(handles, labels, framealpha=0.3, fontsize=12)

                    #if (metric == 'loss') and (ens_attr == "oracle_evals") and (severity == 0):
                        #ax.legend(handles, labels, framealpha=0.3, fontsize=12)

                    if (metric == 'loss') and (ens_attr == 'evals') and \
                       (args.dataset == 'tiny') and ('darts_esa' in args.methods):
                        handles, labels = ax.get_legend_handles_labels()
                        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                        labels_new = tuple(['DeepEns (DARTS)', 'DeepEns+ES (DARTS)', 'DeepEns (AmoebaNet)', 'DeepEns+ES (AmoebaNet)', 'DeepEns (RS)', 'DeepEns+ES (RS)', 
                                            'NES-RS (K = 200)', 'NES-RS (K = 400)', 'NES-RE (K = 200)', 'NES-RE (K = 400)'])
                        idx_old = [labels.index(x) for x in labels_new]
                        labels = labels_new
                        handles = tuple([handles[i] for i in idx_old])
                        ax.legend(handles, labels, framealpha=0.45, fontsize=11, ncol=2, handletextpad=0.1, columnspacing=0.2,
                                  loc='best')
                                  #loc='center left', bbox_to_anchor=(1, 0.5))
                                  #bbox_to_anchor=(0.5, 1.05),
                                  #ncol=2)
                        #ax.set_ylim(top=1.77)

                    #if len(args.Ms) == 1:
                    #    i = axes[-1]
                    #    plt.setp(i, xlim=(PLOT_EVERY, BUDGET))
                    #    plt.setp(i.xaxis.get_majorticklabels(), ha="right")
                    #else:
                    #    for i in axes[-1]:
                    #        plt.setp(i, xlim=(PLOT_EVERY, BUDGET))
                    #        plt.setp(i.xaxis.get_majorticklabels(), ha="right")
                    # for i in axes:
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

elif args.plot_type == "severity":
    for data_type in data_types:
        for ens_attr in ens_attrs:
            for metric in ["loss", "error", "ece", "normalized_disagreement"]:
                if (ens_attr == "disagreement") ^ ("disagreement" in metric):
                    continue
                # fig, axes = plt.subplots(
                #     1,
                #     len(args.Ms),
                #     figsize=(3.5 * len(args.Ms), 6.0),
                #     sharex="col",
                #     sharey=False,
                # )

                for j, M in enumerate(args.Ms):
                    fig, ax = plt.subplots(
                        1,
                        1,
                        figsize=(4., 4.5),
                        sharex="col",
                        sharey=False,
                    )

                    # ax = axes[j]

                    for pool_name in args.methods:
                        y_mean = []
                        y_err = []
                        for i, severity in enumerate(severities):

                            if pool_name in ["nes_re", "nes_rs",
                                             "amoebanet_esa", "darts_esa",
                                             "nes_rs_esa",
                                             "darts_rs", "darts_hyper", "joint"]:
                                xs = []
                                ys = []

                                if pool_name in ["nes_re", "nes_rs"]:
                                    runs = args.runs
                                elif pool_name == "nes_rs_esa":
                                    runs = ["run_3"]
                                else:
                                    runs = ["run_1"]

                                for plot_dir in [os.path.join(args.load_plotting_data_dir, p) for p in runs]:


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

                                label = f"{label_names[pool_name]}"

                                # ax.plot(x, y, label=label, color=colors[pool_name],
                                #         linewidth=2, marker=markers[pool_name],
                                #         markersize=7, markevery=2)
                                # ax.fill_between(x, y - 1.96*err/np.sqrt(len(xs)), y
                                #                 + 1.96*err/np.sqrt(len(xs)),
                                #                 color=colors[pool_name], alpha=.2)
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

                                # ax.plot(x, mean, label=label, color=colors[pool_name],
                                #         linewidth=2, marker=markers[pool_name],
                                #         markersize=7, markevery=2)
                                # ax.fill_between(x, mean -
                                #                 1.96*std/np.sqrt(len(xs)), mean
                                #                 + 1.96*std/np.sqrt(len(xs)),
                                #                 color=colors[pool_name], alpha=.2)

                                y_mean.append(mean[-1])
                                y_err.append(1.96*std[-1]/np.sqrt(len(xs)))
                                # import pdb; pdb.set_trace()

                        color = colors[pool_name]
                        marker = markers[pool_name]

                        y_mean = np.array(y_mean)
                        y_err = np.array(y_err)

                        if "esa" in pool_name: 
                            ls = "dotted" 
                        else:
                            ls = "-"

                        if pool_name == 'deepens_darts_anchor':
                            ls = ':'

                        ax.plot(severities, y_mean, label=label, color=color, marker=marker, linewidth=2, markersize=7, markevery=1, ls=ls) 
                        ax.fill_between(severities, y_mean - y_err, y_mean + y_err, color=color, alpha=0.15)

                        # if i == (len(severities) - 1):
                        ax.set_xlabel("Shift severity", fontsize=17)
                        # if i == 0:
                        ax.set_title(f"{dataset_to_title[args.dataset]}")

                        # if j == 0:
                        # sev_level = (
                        #     "(no shift)" if severity == 0 else f"(severity = {severity})"
                        # )
                        ax.set_ylabel(
                            "{}".format(ens_attr_to_title[ens_attr])
                            + f" {metric_label[metric]}"
                        )
                        if (ens_attr in ["oracle_evals",
                                         "avg_baselearner_evals", "evals"]): # and (args.dataset == 'cifar10'):
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

                    # handles, labels = ax.get_legend_handles_labels()
                    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                    # ax.legend(handles, labels, framealpha=0.6, fontsize=10)
                    #if len(args.Ms) == 1:
                    #    i = axes[-1]
                    #    plt.setp(i, xlim=(PLOT_EVERY, BUDGET))
                    #    plt.setp(i.xaxis.get_majorticklabels(), ha="right")
                    #else:
                    #    for i in axes[-1]:
                    #        plt.setp(i, xlim=(PLOT_EVERY, BUDGET))
                    #        plt.setp(i.xaxis.get_majorticklabels(), ha="right")
                    # for i in axes:
                    if (metric == 'ece') and (ens_attr == "evals") and (args.dataset == "cifar10"):
                        handles, labels = ax.get_legend_handles_labels()
                        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                        if 'deepens_minimum' not in args.methods:
                            labels_new = tuple(['DeepEns (RS)', 'DeepEns (AmoebaNet)',
                                                'DeepEns (DARTS)', 'AnchoredEns (DARTS)',
                                                'NES-RS', 'NES-RE'])
                        else:
                            labels_new = tuple(['DeepEns (RS)', 'DeepEns (GDAS)',
                                                'DeepEns (best arch.)',
                                                'NES-RS', 'NES-RE'])
                        idx_old = [labels.index(x) for x in labels_new]
                        labels = labels_new
                        handles = tuple([handles[i] for i in idx_old])
                        ax.legend(handles, labels, framealpha=0.3, fontsize=12)

                    ax.set_xticks(severities)
                    ax.set_xticklabels(severities)
                    plt.setp(ax, xlim=(0, max(severities)))
                    plt.setp(ax.xaxis.get_majorticklabels(), ha="right")

                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                    Path(os.path.join(SAVE_DIR, data_type, ens_attr)).mkdir(
                        exist_ok=True, parents=True
                    )
                    plt.tight_layout()
                    fig.savefig(
                        os.path.join(SAVE_DIR, data_type, ens_attr, f"metric_{metric}_M_{M}.pdf"),
                        bbox_inches="tight",
                        pad_inches=0.01,
                    )

                    print("Plot saved.")
                    plt.close("all")
