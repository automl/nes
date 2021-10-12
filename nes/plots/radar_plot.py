import os
import argparse
import dill as pickle
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from math import pi
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
import seaborn as sns

from nes.ensemble_selection.config import BUDGET, PLOT_EVERY

matplotlib.use("Agg")

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


def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])


def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1)
                     * (x2 - x1) + x1)
    return sdata


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


class ComplexRadar():
    """
    From: https://datascience.stackexchange.com/questions/6084/how-do-i-create-a-complex-radar-chart
    """
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=4):
        angles = np.arange(0, 360, 360./len(variables)) + 360./len(variables)/2 # added offset to rotate whole thing

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i))
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles,
                                         labels=variables)


        # this doesnt seem to work for polar plots, i.e. it doesn't rotate anything
        # [txt.set_rotation(angle - 180) for txt, angle 
        #      in zip(text, angles)]

        # attempting this instead (actually realized this is done below...)
        labels = []
        for label, angle in zip(axes[0].get_xticklabels(), angles):
            x,y = label.get_position()
            lab = axes[0].text(x,y, label.get_text(), transform=label.get_transform(),
                        ha=label.get_ha(), va=label.get_va())
            
            lab.set_rotation(angle - 90 if angle < 180 else angle + 90)
            labels.append(lab)
        axes[0].set_xticklabels([])


        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2))
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                print(grid)
                #grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            gridlabel[-1] = ""
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])

            for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(10)

            for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(10)

        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, annotate=False, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

# ===================================
# Plot things
# ===================================

metric_label = {"loss": "NLL", "error": "Error", "ece": "ECE"}

colors = {
    "nes_rs": "forestgreen",
    "deepens_rs": "dodgerblue",
    "nes_re": "crimson",
    "deepens_darts": "black",
    "deepens_gdas": "cyan",
    "deepens_minimum": "dodgerblue",
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
    'deepens_minimum': 'DeepEns (best arch.)',
    'deepens_darts': 'DeepEns (DARTS)',
    'deepens_gdas': 'DeepEns (GDAS)',
    'deepens_amoebanet': 'DeepEns (AmoebaNet)',
}

ens_attr_to_title = {
    "evals": "Ensemble",
    "avg_baselearner_evals": "Average baselearner",
    "oracle_evals": "Oracle ensemble",
}

SAVE_DIR = args.save_dir
#data_types = ["test", "val"]
data_type = "test"
ens_attrs = ["evals", "avg_baselearner_evals", "oracle_evals", "disagreement"]
#severities = range(6) if (args.dataset in ["cifar10", "cifar100", "tiny"]) else range(1)
severity = 0
M = 3


data_dict = {
    'methods': [x for x in args.methods],
    'Ensemble NLL': list(np.zeros(len(args.methods))),
    'Error': list(np.zeros(len(args.methods))),
    'ECE': list(np.zeros(len(args.methods))),
    'Avg. base learner NLL': list(np.zeros(len(args.methods))),
    'Oracle NLL': list(np.zeros(len(args.methods))),
    '1 - Pred. Disagr.': list(np.zeros(len(args.methods)))
}

metric_to_dict = {'loss': 'Ensemble NLL', 'error': 'Error', 'ece': 'ECE'}

for ens_attr in ens_attrs:
    for metric in ["loss", "error", "ece",
                   "normalized_disagreement"]:
        if (ens_attr == "disagreement") ^ ("disagreement" in metric):
            continue

        for pool_name in args.methods:
            if pool_name in ["nes_rs", "nes_re"]:
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

                y = np.array(ys).mean(axis=0)
                # plot only the last value
                idx = data_dict['methods'].index(pool_name)
                if (metric in ["loss", "error", "ece"]) and (ens_attr == 'evals'):
                    data_dict[metric_to_dict[metric]][idx] = y[-1]
                elif (metric == 'loss') and (ens_attr ==
                                             'avg_baselearner_evals'):
                    data_dict['Avg. base learner NLL'][idx] = y[-1]
                elif (metric == 'loss') and (ens_attr ==
                                             'oracle_evals'):
                    data_dict['Oracle NLL'][idx] = y[-1]
                elif (metric == 'normalized_disagreement') and (ens_attr ==
                                                                'disagreement'):
                    data_dict['1 - Pred. Disagr.'][idx] = 1 - y[-1]


            elif pool_name in ["deepens_darts", "deepens_pcdarts",
                               "deepens_amoebanet", "deepens_gdas",
                               "deepens_minimum"]:
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

                yy = plotting_data[str(M)][str(severity)][ens_attr][pool_name].y
                y = yy[data_type][str(severity)][metric]
                # plot only the last value
                idx = data_dict['methods'].index(pool_name)
                if (metric in ["loss", "error", "ece"]) and (ens_attr == 'evals'):
                    data_dict[metric_to_dict[metric]][idx] = y
                elif (metric == 'loss') and (ens_attr ==
                                             'avg_baselearner_evals'):
                    data_dict['Avg. base learner NLL'][idx] = y
                elif (metric == 'loss') and (ens_attr ==
                                             'oracle_evals'):
                    data_dict['Oracle NLL'][idx] = y
                elif (metric == 'normalized_disagreement') and (ens_attr ==
                                                                'disagreement'):
                    data_dict['1 - Pred. Disagr.'][idx] = 1 - y

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

                mean = np.mean(all_pools, axis=0)
                # plot only the last value
                idx = data_dict['methods'].index(pool_name)
                if (metric in ["loss", "error", "ece"]) and (ens_attr == 'evals'):
                    data_dict[metric_to_dict[metric]][idx] = mean[-1]
                elif (metric == 'loss') and (ens_attr ==
                                             'avg_baselearner_evals'):
                    data_dict['Avg. base learner NLL'][idx] = mean[-1]
                elif (metric == 'loss') and (ens_attr ==
                                             'oracle_evals'):
                    data_dict['Oracle NLL'][idx] = mean[-1]
                elif (metric == 'normalized_disagreement') and (ens_attr ==
                                                                'disagreement'):
                    data_dict['1 - Pred. Disagr.'][idx] = mean[-1]

categories = list(data_dict.keys())
categories.remove('methods')
N = len(data_dict['methods'])

df = pd.DataFrame(data_dict)

########################
fig = plt.figure(figsize=(6, 6))
#loss, error, ece, avg, dis, orc
ranges = [(1.75, 2.0), (0.45, 0.51), (0.02, 0.035), (1.98, 2.32), (1.28, 1.65), (0., 0.2)]
#ranges = [(2.0, 1.5), (0.51, 0.45), (0.035, 0.02), (1.9, 2.5), (0.8, 1.0)]

radar = ComplexRadar(fig, categories, ranges)

for i in range(N):
    values=df.loc[i].drop('methods').values.flatten().tolist()
    # import pdb; pdb.set_trace()
    # idx_to_modify = categories.index("Pred. Disagr.")
    # values_after_modif = [v if idx != idx_to_modify else 1 - v for idx, v in enumerate(values)]

    radar.plot(values, annotate=False if i==0 else False,
               linewidth=2, color=colors[data_dict['methods'][i]], linestyle='solid',
               marker=markers[data_dict['methods'][i]], markersize=8,
               label=label_names[data_dict['methods'][i]])
    radar.fill(values, color=colors[data_dict['methods'][i]], alpha=.05)

# Put a legend below current axis
radar.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=False, ncol=3)
plt.grid(color='#AAAAAA')
radar.ax.set_facecolor('#FAFAFA')
radar.ax.spines['polar'].set_color('#222222')

# Go through labels and adjust alignment based on where
# it is in the circle.
angles = np.linspace(0,2*np.pi,len(radar.ax.get_xticklabels())+1)
angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
angles = np.rad2deg(angles)

labels = []
for label, angle in zip(radar.ax.get_xticklabels(), angles):
    label.set_horizontalalignment('center')

    x,y = label.get_position()
    lab = radar.ax.text(x,y, label.get_text(), transform=label.get_transform(),
                        ha=label.get_ha(), va=label.get_va())
    if label.get_text() in ['Error', 'Avg. base learner NLL', 'Ensemble NLL']:
        offset = 180
    else:
        offset = 0
    lab.set_rotation(offset + 90 + angle)
    labels.append(lab)
radar.ax.set_xticklabels([])


# add lines
import matplotlib.lines as lines

l = lines.Line2D([0.05, 1.05], [0.55, 0.55], transform=fig.transFigure, figure=fig,
                 color='gray', linewidth=2, linestyle='--')
fig.lines.extend([l])

plt.text(0.99, 0.58, "Performance metrics", size=10, rotation=0.,
         ha="center", va="center", transform=fig.transFigure,
         bbox=dict(boxstyle="round",
                   ec=(1., 0.0, 0.0),
                   fc=(1., 0.9, 0.9),
                   )
         )
plt.text(0.99, 0.52, "Ensemble diagnostics", size=10, rotation=-0.,
         ha="center", va="center", transform=fig.transFigure,
         bbox=dict(boxstyle="round",
                   ec=(1., 0.0, 0.0),
                   fc=(1., 0.9, 0.9),
                   )
         )

Path(os.path.join(SAVE_DIR, data_type)).mkdir(
    exist_ok=True, parents=True
)

plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_DIR, data_type,
                 f"radar_plot.pdf"),
    bbox_inches="tight",
    pad_inches=0.09,
    dpi=100,
)

print("Plot saved.")
plt.close("all")

