from collections import namedtuple

# ======================================
# Some global configs for the experiments.

BUDGET = 400 # Maximum number of networks evaluated.
PLOT_EVERY = 25 # Frequency at which incumbents are chosen (and plotting is done).
MAX_M = 30 # Largest ensemble size used.


model_seeds = namedtuple(typename="model_seeds", field_names=["arch", "init", "scheme"])

dataset_to_budget = {
    "cifar10": 400,
    "cifar100": 400,
    "fmnist": 400,
    "tiny": 200,
    "imagenet": 1000
}


# deepens_rs not included here yet since the archs are the best ones from the sample trained for nes_rs. See rs_incumbets.py
SCHEMES = ["nes_rs", "nes_re", "deepens_darts", "deepens_gdas",
           "nes_rs_oneshot", "nes_re_50k", "nes_rs_darts",
           "deepens_minimum", "nes_rs_50k", "deepens_amoebanet_50k",
           "deepens_darts_50k", "deepens_amoebanet", "darts_esa", "amoebanet_esa", "nes_rs_esa",
           "deepens_darts_anchor", "darts_rs", "darts_hyper", "joint"]

POOLS = {
    scheme: [model_seeds(arch=seed, init=seed, scheme=scheme) for seed in range(BUDGET)]
    for scheme in SCHEMES
    if "nes" in scheme
}

POOLS.update(
    {
        scheme: [model_seeds(arch=0, init=seed, scheme=scheme) for seed in range(MAX_M)]
        for scheme in SCHEMES
        if "deepens" in scheme
    }
)

POOLS.update(
    {
        scheme: [model_seeds(arch=0, init=seed, scheme=scheme) for seed in range(BUDGET)]
        for scheme in SCHEMES
        if scheme in ["darts_esa", "amoebanet_esa"]
    }
)

# tiny seed 3
POOLS.update(
    {
        scheme: [model_seeds(arch=7, init=seed, scheme=scheme) for seed in range(BUDGET)]
        for scheme in SCHEMES
        if scheme == "nes_rs_esa"
    }
)


POOLS.update(
    {
        scheme: [model_seeds(arch=seed, init=seed, scheme=scheme) for seed in range(BUDGET)]
        for scheme in SCHEMES
        if scheme in ["darts_rs", "darts_hyper", "joint"]
    }
)


