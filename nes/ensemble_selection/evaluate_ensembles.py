import os
import dill as pickle
import itertools
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import time
import argparse

from types import SimpleNamespace
from collections import defaultdict, namedtuple
from pathlib import Path

from nes.ensemble_selection.containers import Ensemble, load_baselearner
from nes.ensemble_selection.utils import args_to_device, model_seeds, POOLS
from nes.ensemble_selection.config import MAX_M
from nes.ensemble_selection.rs_incumbents import get_incumbents

parser = argparse.ArgumentParser()
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="Index of GPU device to use. For CPU, set to -1. Default: 0.",
)
parser.add_argument(
    "--esa",
    type=str,
    default="beam_search",
    help="Ensemble selection algorithm. See nes/ensemble_selection/esas.py. Default: beam_search.",
)
parser.add_argument("--M", type=int, default=5, help="Ensemble size. Default: 5.")
parser.add_argument(
    "--save_dir",
    type=str,
    help="Directory to save ensemble evaluation data for eventual plotting.",
)
parser.add_argument(
    "--load_bsls_dir",
    type=str,
    help="Directory where the baselearners in the pool are saved. Will usually depend on --method.",
)
parser.add_argument(
    "--load_ens_chosen_dir",
    type=str,
    help="Directory where output of ensembles_from_pools.py is saved. *Only used when --method is nes_rs or nes_re.*",
)
parser.add_argument(
    "--incumbents_dir",
    type=str,
    help="Directory where output of rs_incumbents.py is saved.",
)
parser.add_argument(
    "--nes_rs_bsls_dir",
    type=str,
    help="Directory where nes_rs baselearners are saved. *Only used when --method is deepens_rs.*",
)
parser.add_argument(
    "--method",
    choices=["nes_rs", "nes_re", "deepens_rs", "deepens_darts", "deepens_amoebanet"],
    type=str,
)
parser.add_argument(
    "--dataset", choices=["cifar10", "fmnist"], type=str, help="Dataset."
)

args = parser.parse_args()

torch.cuda.set_device(args_to_device(args.device))


def evaluated_ensemble(list_of_bsl_ids):

    baselearners = [library[m] for m in list_of_bsl_ids]
    for b in baselearners:
        b.to_device(args_to_device(args.device))

    ensemble = Ensemble(baselearners)

    ensemble.compute_preds()
    ensemble.compute_evals()
    ensemble.preds = None  # clear memory

    ensemble.compute_avg_baselearner_evals()

    ensemble.compute_oracle_preds()
    ensemble.compute_oracle_evals()
    ensemble.oracle_preds = None  # clear memory

    for b in baselearners:
        b.to_device(args_to_device(-1))

    torch.cuda.empty_cache()

    return ensemble


def read_txt_lines(save_dir):
    with open(save_dir, "r") as f:
        lines = f.readlines()
    return lines


SAVE_DIR = args.save_dir

if args.method != "deepens_rs":
    models_to_load = POOLS[args.method]
else:
    incumbent_archs = [int(k) for k in read_txt_lines(args.incumbents_dir)]
    models_to_load = [
        model_seeds(arch=k1, init=k2, scheme="deepens_rs")
        for k1 in incumbent_archs
        for k2 in range(MAX_M)
    ]

    incumbents_dct = get_incumbents(
        pool_name="nes_rs",  # use the same sample of archs as the one used for nes_rs
        load_bsls_dir_name=args.nes_rs_bsls_dir,
        dataset=args.dataset,
    )


library = {
    k: load_baselearner(
        model_id=k,
        load_nn_module=False,
        working_dir=os.path.join(args.load_bsls_dir, f"run_{k.arch}"),
    )
    for k in models_to_load
}

Coordinates = namedtuple("Coordinates", field_names=["x", "y"])

severities = range(6) if (args.dataset == "cifar10") else range(1)

nested_dict = lambda: defaultdict(nested_dict)

ens_attrs = ["evals", "avg_baselearner_evals", "oracle_evals"]

plotting_data = nested_dict()

if args.method in ["nes_rs", "nes_re"]:
    with open(
        os.path.join(
            args.load_ens_chosen_dir,
            f"ensembles_chosen__esa_{args.esa}_M_{args.M}_pool_{args.method}.pickle",
        ),
        "rb",
    ) as f:
        ens_chosen_dct = pickle.load(f)

    for severity in severities:
        ens_chosen = ens_chosen_dct["ensembles_chosen"][str(severity)]

        x = ens_chosen_dct["num_arch_samples"]
        yy = [evaluated_ensemble(bsls) for bsls in ens_chosen]

        assert len(x) == len(yy)

        for ens_attr in ens_attrs:
            y = [getattr(e, ens_attr) for e in yy]
            plotting_data[str(args.M)][str(severity)][ens_attr][args.esa][
                args.method
            ] = Coordinates(x=x, y=y)

        print(f"Done severity = {severity}")

    Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)
    with open(
        os.path.join(
            SAVE_DIR,
            f"plotting_data__esa_{args.esa}_M_{args.M}_pool_{args.method}.pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(plotting_data, f)

elif args.method in ["deepens_darts", "deepens_amoebanet"]:

    for severity in severities:

        bsls = models_to_load[: args.M]
        yy = evaluated_ensemble(bsls)

        for ens_attr in ens_attrs:
            y = getattr(yy, ens_attr)
            plotting_data[str(args.M)][str(severity)][ens_attr][
                args.method
            ] = Coordinates(x=None, y=y)

        print(f"Done severity = {severity}")

    Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)
    with open(
        os.path.join(
            SAVE_DIR, f"plotting_data__M_{args.M}_pool_{args.method}.pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(plotting_data, f)

elif args.method == "deepens_rs":

    def get_incumbent_bsl(incumbents_dct, severity):
        bsls_dct = incumbents_dct[str(severity)]

        k, incumbent = list(bsls_dct.items())[0]

        output_dct = {k: incumbent}
        for k, v in bsls_dct.items():
            if v == incumbent:
                continue
            else:
                incumbent = v
                output_dct.update({k: v})

        return output_dct

    def incumbent_to_ensemble_bsls_list(model_id, model_lib, M):
        rand_inits_bsls = [k for k in model_lib if k.arch == model_id.arch]
        assert (
            len(rand_inits_bsls) == MAX_M
        )  # we train MAX_M random inits assuming the largest ensemble is of size MAX_M
        return rand_inits_bsls[:M]

    for severity in severities:
        incumbents = get_incumbent_bsl(incumbents_dct, severity)
        x = [int(k) for k in incumbents.keys()]

        incumbent_ens = [
            incumbent_to_ensemble_bsls_list(bsl_ls[0], models_to_load, args.M)
            for bsl_ls in incumbents.values()
        ]
        yy = [evaluated_ensemble(bsls) for bsls in incumbent_ens]

        assert len(x) == len(yy)

        for ens_attr in ens_attrs:
            y = [getattr(e, ens_attr) for e in yy]
            plotting_data[str(args.M)][str(severity)][ens_attr][
                args.method
            ] = Coordinates(x=x, y=y)

        print(f"Done severity = {severity}")

    Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)
    with open(
        os.path.join(
            SAVE_DIR, f"plotting_data__M_{args.M}_pool_{args.method}.pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(plotting_data, f)


print("Ensemble evaluation completed.")

