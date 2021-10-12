import sys
import os

from nes.optimizers.baselearner_train.utils import parse_config
from nes.optimizers.cluster_worker import REWorker

run_id = sys.argv[1]

path = os.path.join('experiments-nips21/cifar10/baselearners/nes_re', run_id,
                    "configs.json")

with open(path) as f:
    configs = [eval(x[:-1]) for x in f.readlines()]

config_space = REWorker.get_configspace()

save_dir = os.path.join('experiments-nips21/cifar10/baselearners/nes_re',
                        run_id,
                        'sampled_configs')

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

for config in configs:
    config_id = config[0][0]
    config_arch = config[1]

    genotype = parse_config(config_arch, config_space)
    print(config_id)
    print(genotype)

    with open(os.path.join(save_dir, "arch_%d.txt"%config_id), "w") as f:
        f.write("%s"%(str(genotype)))
