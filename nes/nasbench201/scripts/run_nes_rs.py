import os
import random
import warnings
import numpy as np
import pickle
warnings.filterwarnings('ignore')

from nes.nasbench201 import NB201Worker as Worker
from nes.ensemble_selection.config import BUDGET


def rs_master(args):
    assert args.global_seed > 0, "global seed should be greater than 0"

    # set seed relative to the global seed and fix working directory
    seed_id = (args.global_seed - 1) * BUDGET + args.arch_id
    args.working_directory = os.path.join(args.working_directory,
                                          'run_%d'%args.global_seed)

    # generate a random architecture
    np.random.seed(seed_id)
    random.seed(seed_id)

    # load list of possible ids
    with open('nes/utils/nb201/configs/{}.pkl'.format(args.dataset), 'rb') as f:
        list_of_ids = pickle.load(f)
    nb201_id = str(random.choice(list_of_ids))
    print(nb201_id)
    genotype = '0'*(6-len(nb201_id)) + nb201_id

    genotype_save_foldername = os.path.join(args.working_directory,
                                            'random_archs')
    if not os.path.exists(genotype_save_foldername):
        os.makedirs(genotype_save_foldername, exist_ok=True)

    with open(os.path.join(genotype_save_foldername,
                           'arch_%d.txt'%args.arch_id), 'w') as f:
        to_save = genotype if type(genotype) is str else str(genotype)
        f.write('%s'%(to_save))

    # no need for a master node in the NES-RS case. Just instantiate a worker
    # that trains the architectures and computes predictions on clean and
    # shifted data
    worker = Worker(working_directory=args.working_directory,
                    run_id=args.arch_id,
                    scheme=args.scheme,
                    n_datapoints=args.n_datapoints,
                    dataset=args.dataset)

    worker.compute(config=genotype,
                   budget=200,
                   config_id=args.arch_id,
                   seed_id=args.seed_id)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_directory', type=str,
                        help='directory where the generated results are saved')
    parser.add_argument('--arch_id', type=int, default=0,
                        help='architecture id number')
    parser.add_argument('--seed_id', type=int, default=0,
                        help='seed number')
    parser.add_argument('--global_seed', type=int, default=1,
                        help='global seed number')
    parser.add_argument('--dataset', type=str, default='tiny',
                        help='image dataset')
    parser.add_argument('--scheme', type=str, default='nes_rs',
                        help='scheme name, i.e. nes or deepens variants')
    parser.add_argument('--n_datapoints', type=int, default=40000,
                        help='train size')
    args = parser.parse_args()

    rs_master(args)
