import os
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from nes.optimizers.baselearner_train.utils import sample_random_genotype
from nes.optimizers.cluster_worker import REWorker as Worker


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_directory', type=str,
                        help='directory where the generated results are saved')
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
    parser.add_argument('--scheme', type=str, default='nes_rs',
                        help='scheme name, i.e. nes or deepens variants')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode: run for a single mini-batch')
    args = parser.parse_args()

    # generate a random architecture
    np.random.seed(args.arch_id)
    genotype = sample_random_genotype(steps=4, multiplier=4)
    genotype_save_foldername = os.path.join(args.working_directory,
                                            'random_archs')

    if not os.path.exists(genotype_save_foldername):
        os.makedirs(genotype_save_foldername, exist_ok=True)

    with open(os.path.join(genotype_save_foldername,
                           'arch_%d.txt'%args.arch_id), 'w') as f:
        f.write('%s'%(str(genotype)))

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

