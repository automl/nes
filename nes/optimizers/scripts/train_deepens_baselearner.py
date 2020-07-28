import os
import warnings
warnings.filterwarnings('ignore')

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
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of epochs to train the baselearner')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Mini-batch size to train the baselearner')
    parser.add_argument('--dataset', type=str, default='fmnist',
                        help='image dataset')
    parser.add_argument('--scheme', type=str, default='deepens_rs',
                        help='scheme name, i.e. nes or deepens variants')
    mutex.add_argument('--train_darts', action='store_true', default=False,
                        help='debug mode: run for a single mini-batch')
    mutex.add_argument('--train_amoebanet', action='store_true', default=False,
                        help='debug mode: run for a single mini-batch')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode: run for a single mini-batch')
    args = parser.parse_args()

    # load either DARTS, AmoebaNet, or the incumbent architectures from NES-RS
    if args.train_darts:
        genotype = DARTS
    elif args.train_amoebanet:
        genotype = AmoebaNet
    else:
        with open(os.path.join(args.arch_path, 'arch_%d.txt'%args.arch_id), 'r') as f:
            genotype = eval(f.read())

    # no need for a master node in the NES-RS case. Just instantiate a worker
    # that trains the architectures and computes predictions on clean and
    # shifted data
    worker = Worker(working_directory=args.working_directory,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    run_id=args.arch_id,
                    scheme=args.scheme,
                    dataset=args.dataset,
                    debug=args.debug)

    worker.compute(genotype,
                   budget=args.num_epochs,
                   config_id=(args.arch_id, 0, 0),
                   seed_id=args.seed_id)

