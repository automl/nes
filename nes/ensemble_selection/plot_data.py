import argparse
import dill as pickle
import matplotlib.pyplot as plt
import matplotlib

from nes.ensemble_selection.config import BUDGET, PLOT_EVERY

matplotlib.use("Agg")
import os
from pathlib import Path

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
    "--dataset", choices=["cifar10", "fmnist"], type=str, help="Dataset."
)


args = parser.parse_args()

SAVE_DIR = args.save_dir
data_type = "test"
ens_attrs = ["evals", "avg_baselearner_evals", "oracle_evals"]
severities = range(6) if (args.dataset == "cifar10") else range(1)

# ===================================
# Plot things
# ===================================

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
plt.rcParams["font.size"] = 16


metric_label = {"loss": "NLL", "error": "Error", "ece": "ECE"}

colors = {
    "nes_rs": "forestgreen",
    "deepens_rs": "dodgerblue",
    "nes_re": "crimson",
    "deepens_darts": "black",
    "deepens_amoebanet": "darkorange",
}

ens_attr_to_title = {
    "evals": "Ensemble",
    "avg_baselearner_evals": "Average baselearner",
    "oracle_evals": "Oracle ensemble",
}

for ens_attr in ens_attrs:
    for metric in ["loss", "error", "ece"]:
        fig, axes = plt.subplots(
            len(severities),
            len(args.Ms),
            figsize=(5.0 * len(args.Ms), 5.0 * len(severities)),
            sharex="col",
            sharey=False,
        )

        if len(severities) == 1:
            axes = [axes]
        for i, severity in enumerate(severities):
            for j, M in enumerate(args.Ms):
                # import pdb; pdb.set_trace()
                ax = axes[i][j]
                for pool_name in args.methods:
                    if "nes" in pool_name:

                        with open(
                            os.path.join(
                                args.load_plotting_data_dir,
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

                        label = f"{pool_name}"

                        ax.plot(x, y, label=label, color=colors[pool_name])

                    elif pool_name in ["deepens_darts", "deepens_amoebanet"]:
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

                        label = f"{pool_name}"

                        ax.axhline(
                            y,
                            label=label,
                            linestyle="--",
                            color=colors[pool_name],
                            linewidth=2,
                        )

                    elif pool_name in ["deepens_rs"]:
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
                        y = [item[data_type][str(severity)][metric] for item in yy]

                        # extend line until end of plot.
                        x = x + [BUDGET]
                        y = y + [y[-1]]

                        label = f"{pool_name}"

                        ax.plot(x, y, label=label, color=colors[pool_name])

                if i == (len(severities) - 1):
                    ax.set_xlabel("Number of networks evaluated")
                if i == 0:
                    ax.set_title(f"M = {M}")
                if j == 0:
                    sev_level = (
                        "(no shift)" if severity == 0 else f"(severity = {severity})"
                    )
                    ax.set_ylabel(
                        "{}".format(ens_attr_to_title[ens_attr])
                        + "\n"
                        + f"{metric_label[metric]} {sev_level}"
                    )

        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels, framealpha=0.6, fontsize=10)
        for i in axes[-1]:
            plt.setp(i, xlim=(PLOT_EVERY, BUDGET))
            plt.setp(i.xaxis.get_majorticklabels(), ha="right")

        Path(os.path.join(SAVE_DIR, data_type, ens_attr)).mkdir(
            exist_ok=True, parents=True
        )
        plt.tight_layout()
        fig.savefig(
            os.path.join(SAVE_DIR, data_type, ens_attr, f"metric_{metric}.pdf"),
            bbox_inches="tight",
            pad_inches=0.05,
        )

        print("Plot saved.")
        plt.close("all")

