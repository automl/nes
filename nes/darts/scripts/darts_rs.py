import os
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from nes.darts.baselearner_train.utils import sample_random_genotype
from nes.darts.baselearner_train.genotypes import Genotype, DARTS
from nes.darts.cluster_worker import DARTSWorker as Worker
from nes.ensemble_selection.config import BUDGET


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
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train the baselearner')
    parser.add_argument('--lr', type=float, default=0.025,
                        help='leaning rate to train the baselearner')
    parser.add_argument('--wd', type=float, default=3e-4,
                        help='weight decay to train the baselearner')
    parser.add_argument('--n_layers', type=int, default=8,
                        help='Mini-batch size to train the baselearner')
    parser.add_argument('--init_channels', type=int, default=16,
                        help='Mini-batch size to train the baselearner')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Mini-batch size to train the baselearner')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='image dataset')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        help='scheme name, i.e. nes or deepens variants')
    parser.add_argument('--scheme', type=str, default='darts_rs',
                        help='scheme name, i.e. nes or deepens variants')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of CPU workers')
    parser.add_argument('--nes_cell', action='store_true', default=False,
                        help="randomly sample cells")
    parser.add_argument('--nes_depth_width', action='store_true', default=False,
                        help="randomly sample channels and cell number")
    parser.add_argument('--hyperensemble', action='store_true', default=False,
                        help="randomly sample lr and wd")
    parser.add_argument('--grad_clip', action='store_true', default=False,
                        help='debug mode: run for a single mini-batch')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode: run for a single mini-batch')
    args = parser.parse_args()

    assert args.global_seed > 0, "global seed should be greater than 0"

    # set seed relative to the global seed and fix working directory
    arch_id = (args.global_seed - 1) * BUDGET + args.arch_id
    args.working_directory = os.path.join(args.working_directory,
                                          'run_%d'%args.global_seed)

    # generate a random architecture
    np.random.seed(arch_id)
    random.seed(arch_id)
    genotype = DARTS

    if args.hyperensemble:
        #args.lr = random.uniform(0.001, 0.1)
        #args.wd = random.uniform(3e-5, 3e-3)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('lr', lower=0.001,
                                                             upper=0.1,
                                                             log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('wd', lower=3e-5,
                                                             upper=3e-3,
                                                             log=True))
        hypers = cs.sample_configuration().get_dictionary()
        args.lr = hypers['lr']
        args.wd = hypers['wd']
        to_save = (args.lr, args.wd)
    if args.nes_cell:
        genotype = sample_random_genotype(steps=4, multiplier=4)
        to_save = genotype if type(genotype) is str else str(genotype)
    if args.nes_depth_width:
        args.n_layers = random.choice([5, 8, 11])
        args.init_channels = random.choice([12, 14, 16, 18, 20])
        to_save_1 = (args.n_layers, args.init_channels)

    print(args)

    # save the sampled architectures or hyperparameters
    genotype_save_foldername = os.path.join(args.working_directory,
                                            'random_archs')
    hyper_save_foldername = os.path.join(args.working_directory,
                                         'random_hyper')

    if args.nes_cell:
        if not os.path.exists(genotype_save_foldername):
            os.makedirs(genotype_save_foldername, exist_ok=True)

        with open(os.path.join(genotype_save_foldername,
                               'arch_%d.txt'%args.arch_id), 'w') as f:
            f.write('{}'.format(to_save))

    if args.nes_depth_width:
        if not os.path.exists(genotype_save_foldername):
            os.makedirs(genotype_save_foldername, exist_ok=True)

        with open(os.path.join(genotype_save_foldername,
                               'depth_width_%d.txt'%args.arch_id), 'w') as f:
            f.write('{}'.format(to_save_1))

    if args.hyperensemble:
        if not os.path.exists(hyper_save_foldername):
            os.makedirs(hyper_save_foldername, exist_ok=True)

        with open(os.path.join(hyper_save_foldername,
                               'hyper_%d.txt'%args.arch_id), 'w') as f:
            f.write('{}'.format(to_save))


    # no need for a master node in the NES-RS case. Just instantiate a worker
    # that trains the architectures and computes predictions on clean and
    # shifted data
    worker = Worker(working_directory=args.working_directory,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    wd=args.wd,
                    run_id=args.arch_id,
                    scheme=args.scheme,
                    dataset=args.dataset,
                    nb201=False,
                    n_workers=args.n_workers,
                    debug=args.debug)

    worker.compute(genotype,
                   budget=args.num_epochs,
                   config_id=(args.arch_id, 0, 0),
                   seed_id=args.seed_id,
                   global_seed=args.global_seed,
                   grad_clip=args.grad_clip,
                   n_layers=args.n_layers,
                   init_channels=args.init_channels,
                   scheduler=args.scheduler)

