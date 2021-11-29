import os
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from nes.optimizers.baselearner_train.genotypes import Genotype, DARTS, AmoebaNet
from nes.optimizers.cluster_worker import REWorker as Worker


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    mutex = parser.add_mutually_exclusive_group()
    parser.add_argument('--working_directory', type=str,
                        help='directory where the generated results are saved')
    parser.add_argument('--arch_path', type=str, default=None,
                        help='directory where the architecture genotypes are')
    parser.add_argument('--arch_id', type=int, default=0,
                        help='architecture id number')
    parser.add_argument('--seed_id', type=int, default=0,
                        help='seed number')
    parser.add_argument('--global_seed', type=int, default=1,
                        help='global seed number')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs to train the baselearner')
    parser.add_argument('--lr', type=float, default=0.025,
                        help='leaning rate to train the baselearner')
    parser.add_argument('--wd', type=float, default=3e-4,
                        help='weight decay to train the baselearner')
    parser.add_argument('--n_layers', type=int, default=14,
                        help='Mini-batch size to train the baselearner')
    parser.add_argument('--init_channels', type=int, default=48,
                        help='Mini-batch size to train the baselearner')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size to train the baselearner')
    parser.add_argument('--dataset', type=str, default='tiny',
                        help='image dataset')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        help='scheme name, i.e. nes or deepens variants')
    parser.add_argument('--scheme', type=str, default='deepens_rs',
                        help='scheme name, i.e. nes or deepens variants')
    mutex.add_argument('--train_darts', action='store_true', default=False,
                        help='evaluate the arch found by DARTS')
    mutex.add_argument('--train_gdas', action='store_true', default=False,
                        help='evaluate the arch found by GDAS')
    mutex.add_argument('--train_global_optima', action='store_true', default=False,
                        help='evaluate the best architecture in nb201')
    mutex.add_argument('--train_amoebanet', action='store_true', default=False,
                        help='evaluate the arch found by RE')
    parser.add_argument('--nb201', action='store_true', default=False,
                        help='NAS-bench-201 space')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of CPU workers')
    parser.add_argument('--grad_clip', action='store_true', default=False,
                        help='debug mode: run for a single mini-batch')
    parser.add_argument('--anchor', action='store_true', default=False,
                        help='anchored ensembles')
    parser.add_argument('--anch_coeff', type=float, default=1.0,
                        help='anchored ensembles regularization')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode: run for a single mini-batch')
    args = parser.parse_args()

    assert args.global_seed > 0, "global seed should be greater than 0"

    # set seed relative to the global seed and fix working directory
    arch_id = args.arch_id
    args.working_directory = os.path.join(args.working_directory,
                                          'run_%d'%args.global_seed)

    print(args)

    # load either DARTS, AmoebaNet, or the incumbent architectures from NES-RS
    opt_to_id = {
        'cifar10' : {'DARTS'    : '001835', # also in c100 and img
                     'GDAS'     : '003928', # also in c100 and img
                     'Optima'   : '014174'},
        'cifar100': {'DARTS'    : '002057', # also 561, 12830, 3521
                     'GDAS'     : '003203', # also in img
                     'Optima'   : '013934'},
        'imagenet': {'DARTS'    : '004771', # also 1728
                     'GDAS'     : '003928', # also in c100 and c10
                     'Optima'   : '003621'},
    }

    if args.nb201:
        assert not args.train_amoebanet, "Cannot evaluate AmoebaNet on NB201!"
        if args.train_darts:
            genotype = opt_to_id[args.dataset]['DARTS']
        elif args.train_gdas:
            genotype = opt_to_id[args.dataset]['GDAS']
        elif args.train_global_optima:
            genotype = opt_to_id[args.dataset]['Optima']
        else:
            with open(os.path.join(args.arch_path, 'arch_%d.txt'%args.arch_id), 'r') as f:
                genotype = f.read()
    else:
        assert not args.train_gdas, "GDAS not implemented yet!"
        if args.train_darts:
            genotype = DARTS
        elif args.train_amoebanet:
            genotype = AmoebaNet
        else:
            with open(os.path.join(args.arch_path, 'arch_%d.txt'%args.arch_id), 'r') as f:
                genotype = eval(f.read())

        # generate a random architecture
        np.random.seed(arch_id)
        random.seed(arch_id)

        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('lr', lower=0.001,
                                                             upper=0.1,
                                                             log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('wd',
                                                             lower=3e-10,
                                                             upper=3e-4,
                                                             log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('anchor',
                                                             lower=1e-3,
                                                             upper=1.0,
                                                             log=True))
        hypers = cs.sample_configuration().get_dictionary()
        args.lr = hypers['lr']
        args.wd = hypers['wd']
        args.anch_coeff = hypers['anchor']
        to_save = (args.lr, args.wd, args.anch_coeff)

        hyper_save_foldername = os.path.join(args.working_directory,
                                             'random_hyper')
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
                    run_id=args.arch_id,
                    scheme=args.scheme,
                    dataset=args.dataset,
                    nb201=args.nb201,
                    n_workers=args.n_workers,
                    lr=args.lr,
                    wd=args.wd,
                    anchor=args.anchor,
                    anch_coeff=args.anch_coeff,
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

