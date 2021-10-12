"""This finds the incumbents from the nes_rs sampled base learners.
"""

from nes.ensemble_selection.utils import (
    model_seeds,
    args_to_device,
)
from nes.ensemble_selection.config import BUDGET, PLOT_EVERY
from nes.ensemble_selection.utils import POOLS
from nes.ensemble_selection.containers import load_baselearner
import dill as pickle
import os
import argparse
from pathlib import Path


def get_incumbents(pool_name, load_bsls_dir_name, dataset):
    """
    Finds the best model (architecture) by validation loss, i.e. the incumbent.
    """
    DEVICE_NUM = -1  # gpu not needed

    pool_keys = POOLS[pool_name]

    pool = {
        k: load_baselearner(
            model_id=k,
            load_nn_module=False,
            working_dir=os.path.join(load_bsls_dir_name, f"run_{k.arch}"),
        )
        for k in pool_keys
    }
    for model in pool.values():  # move everything to a gpu
        model.to_device(args_to_device(DEVICE_NUM))

    dct = {}
    severities = range(6) if (dataset in ["cifar10", "cifar100", "tiny"]) else range(1)
    for severity in severities:
        print('severity %d'%severity)
        best_models = {}
        for x in range(PLOT_EVERY, BUDGET + 1, PLOT_EVERY):
            current_pool = {k: v for k, v in list(pool.items())[:x]}

            best_model = sorted(
                current_pool,
                key=lambda x: current_pool[x].evals["val"][str(severity)]["loss"],
            )[:1]
            best_models[str(x)] = best_model
            print(best_model)

        dct[str(severity)] = best_models

    return dct


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save .txt file containing incumbent IDs.",
    )
    parser.add_argument(
        "--load_bsls_dir",
        type=str,
        help="Directory where the baselearners in the pool are saved. Will usually depend on --pool_name.",
    )
    parser.add_argument(
        "--pool_name",
        type=str,
        default="nes_rs",
        help="Pool to choose incumbents from. Currently only applied to nes_rs pool for deepens_rs baseline. Default: nes_rs.",
    )
    parser.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "fmnist", "imagenet", "tiny"], type=str, help="Dataset."
    )

    args = parser.parse_args()

    lss = []
    for pool_name in [args.pool_name]:
        incumbents = get_incumbents(pool_name, args.load_bsls_dir, args.dataset)
        for dct in incumbents.values():
            print('###########')
            print(dct['200'])
            for i in dct.values():
                lss.extend(i)

    incumbents_archs = [str(t) for t in sorted([(x.arch) for x in list(set(lss))])]

    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    with open(os.path.join(args.save_dir, "incumbents.txt"), "w") as f:
        for arch_idx in incumbents_archs:
            f.write(arch_idx + "\n")

    print(f"Incumbent architectures found and saved.")

