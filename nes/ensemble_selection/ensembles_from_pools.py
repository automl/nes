import os
import dill as pickle
import torch
import time
import argparse
from collections import defaultdict
from pathlib import Path

from nes.ensemble_selection.utils import (
    model_seeds,
    args_to_device,
    POOLS,
    SCHEMES,
)
from nes.ensemble_selection.config import BUDGET, PLOT_EVERY
from nes.ensemble_selection.containers import Ensemble, Baselearner, load_baselearner
from nes.ensemble_selection.esas import registry as esas_registry, run_esa


# ===============================

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
    "--pool_name",
    type=str,
    default="nes_rs",
    choices=["nes_rs", "nes_re"],
    help="Pool of baselearners used for ensemble selection. Default: nes_rs.",
)
parser.add_argument(
    "--save_dir",
    type=str,
    help="Directory to save data for which ensembles were selected.",
)
parser.add_argument(
    "--load_bsls_dir",
    type=str,
    help="Directory where the baselearners in the pool are saved. Will usually depend on --pool_name.",
)
parser.add_argument(
    "--dataset", choices=["cifar10", "fmnist"], type=str, help="Dataset."
)

args = parser.parse_args()

torch.cuda.set_device(args_to_device(args.device))

SAVE_DIR = args.save_dir

Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

# ===============================


def get_pools_and_num_arch_samples(pool):
    pool_keys = POOLS[pool]
    if args.pool_name not in ["deepens_darts", "deepens_amoeba"]:
        assert len(pool_keys) == BUDGET

        num_arch_samples = range(PLOT_EVERY, BUDGET + 1, PLOT_EVERY)
        pool_at_samples = [pool_keys[:num_samples] for num_samples in num_arch_samples]
    else:
        num_arch_samples = range(1)
        pool_at_samples = [pool_keys]

    return {"pools_at_samples": pool_at_samples, "num_arch_samples": num_arch_samples}


# ===============================

pool_keys = POOLS[args.pool_name]

pool = {
    k: load_baselearner(
        model_id=k,
        load_nn_module=False,
        working_dir=os.path.join(args.load_bsls_dir, f"run_{k.arch}"),
    )
    for k in pool_keys
}
print("Loaded baselearners")

for model in pool.values():  # move everything to right device
    model.to_device(args_to_device(args.device))

pools, num_arch_samples = get_pools_and_num_arch_samples(args.pool_name).values()

esa = esas_registry[args.esa]
severities = range(6) if (args.dataset == "cifar10") else range(1)


result = defaultdict(list)

for i, pool_ids in enumerate(pools):
    for severity in severities:
        print(f"Severity: {severity}")
        population = {k: pool[k] for k in pool_ids}

        ens_chosen = run_esa(
            M=args.M, population=population, esa=esa, val_severity=severity
        )

        result[str(severity)].append(ens_chosen)

    print(
        f"Done {i+1}/{len(pools)} for {args.pool_name}, M = {args.M}, esa = {args.esa}, device = {args.device}."
    )

to_dump = {"ensembles_chosen": result, "num_arch_samples": num_arch_samples}

with open(
    os.path.join(
        SAVE_DIR,
        f"ensembles_chosen__esa_{args.esa}_M_{args.M}_pool_{args.pool_name}.pickle",
    ),
    "wb",
) as f:
    pickle.dump(to_dump, f)

print("Ensemble selection completed.")

