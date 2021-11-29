import os
import random
import warnings
import numpy as np
import pickle
warnings.filterwarnings('ignore')

from nes.darts.baselearner_train.utils import sample_random_genotype
from nes.darts.cluster_worker import REWorker as Worker
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
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size to train the baselearner')
    parser.add_argument('--lr', type=float, default=0.025,
                        help='leaning rate to train the baselearner')
    parser.add_argument('--wd', type=float, default=3e-4,
                        help='weight decay to train the baselearner')
    parser.add_argument('--n_layers', type=int, default=14,
                        help='Mini-batch size to train the baselearner')
    parser.add_argument('--init_channels', type=int, default=48,
                        help='Mini-batch size to train the baselearner')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        help='scheme name, i.e. nes or deepens variants')
    parser.add_argument('--grad_clip', action='store_true', default=False,
                        help='debug mode: run for a single mini-batch')
    parser.add_argument('--dataset', type=str, default='tiny',
                        help='image dataset')
    parser.add_argument('--scheme', type=str, default='nes_rs',
                        help='scheme name, i.e. nes or deepens variants')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of CPU workers')
    parser.add_argument('--n_datapoints', type=int, default=40000,
                        help='train size')
    parser.add_argument('--full_train', action='store_true', default=False,
                        help='debug mode: run for a single mini-batch')
    parser.add_argument('--only_predict', action='store_true', default=False,
                        help='use saved model to evaluate')
    parser.add_argument('--oneshot', action='store_true', default=False,
                        help='use oneshot model to evaluate')
    parser.add_argument('--saved_model', type=str, default=None,
                        help='directory where the oneshot model is')
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
    genotype_save_foldername = os.path.join(args.working_directory,
                                            'random_archs')

    genotype = sample_random_genotype(steps=4, multiplier=4)

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
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    run_id=args.arch_id,
                    scheme=args.scheme,
                    dataset=args.dataset,
                    lr=args.lr,
                    wd=args.wd,
                    n_workers=args.n_workers,
                    n_datapoints=args.n_datapoints,
                    nb201=False,
                    full_train=args.full_train,
                    oneshot=args.oneshot,
                    only_predict=args.only_predict,
                    saved_model=args.saved_model,
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

