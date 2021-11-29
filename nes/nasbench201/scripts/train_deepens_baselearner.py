import os
import warnings
warnings.filterwarnings('ignore')

from nes.nasbench201 import NB201Worker as Worker


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
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='image dataset')
    parser.add_argument('--scheme', type=str, default='deepens_rs',
                        help='scheme name, i.e. nes or deepens variants')
    mutex.add_argument('--train_darts', action='store_true', default=False,
                        help='evaluate the arch found by DARTS')
    mutex.add_argument('--train_gdas', action='store_true', default=False,
                        help='evaluate the arch found by GDAS')
    mutex.add_argument('--train_global_optima', action='store_true', default=False,
                        help='evaluate the best architecture in nb201')
    parser.add_argument('--n_datapoints', type=int, default=40000,
                        help='train size')
    args = parser.parse_args()

    assert args.global_seed > 0, "global seed should be greater than 0"

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

    if args.train_darts:
        genotype = opt_to_id[args.dataset]['DARTS']
    elif args.train_gdas:
        genotype = opt_to_id[args.dataset]['GDAS']
    elif args.train_global_optima:
        genotype = opt_to_id[args.dataset]['Optima']
    else:
        with open(os.path.join(args.arch_path, 'arch_%d.txt'%args.arch_id), 'r') as f:
            genotype = f.read()

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

