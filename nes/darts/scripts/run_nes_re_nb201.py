import os
import random
import warnings
import numpy as np
import pickle
import copy
warnings.filterwarnings('ignore')

from nes.optimizers.baselearner_train.utils import sample_random_genotype
from nes.optimizers.cluster_worker import REWorker as Worker

from nes.ensemble_selection.config import BUDGET
from nes.ensemble_selection.containers import load_baselearner
from nes.ensemble_selection.esas import run_esa
from nes.ensemble_selection.esas import registry as esas_registry
from nes.ensemble_selection.utils import model_seeds


OPS = [
    'none',
    'skip_connect',
    'avg_pool_3x3',
    'nor_conv_1x1',
    'nor_conv_3x3',
]


get_genotype = lambda x: '0'*(6-len(x)) + x
get_arch_str = lambda op_list: '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*op_list)


def select_new_parent(population, sample_size, scheme, working_directory, esa,
                      severity_list):
    population_baselearners = {
        x[2]: load_baselearner(model_id=model_seeds(x[2], x[2], scheme),
                               load_nn_module=False,
                               working_dir=os.path.join(working_directory,
                                                        'run_'+str(x[2])))
        for x in population
    }
    severity = random.choice([int(x) for x in severity_list.split()])

    pop_sample = run_esa(M=sample_size,
                         population=population_baselearners,
                         esa=esas_registry[esa],
                         val_severity=severity)
    print('ESA: {}'.format(pop_sample))

    parent_id = random.choice(pop_sample)
    parent_baselearner = list(filter(lambda x: x[2] == parent_id,
                                     population))
    assert len(parent_baselearner) == 1

    return parent_baselearner[0]


def mutate_arch(architecture):
    # no hidden state mutation since the cell is fully connected
    # following https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py#L91
    # we also remove the 'identity' mutation 
    mutation = np.random.choice(['op_mutation'])
    parent = copy.deepcopy(architecture[1]) # work with the list of ops only

    if mutation == 'identity':
        return parent
    elif mutation == 'op_mutation':
        edge_id = random.randint(0, len(parent)-1)
        edge_op = parent[edge_id]
        sampled_op = random.choice(OPS)
        while sampled_op == edge_op:
            sampled_op = random.choice(OPS)
        parent[edge_id] = sampled_op
        return parent


def sample_random_genotype():
    sampled_ops = np.random.choice(OPS, 6)
    arch_str = get_arch_str(sampled_ops)
    return arch_str, sampled_ops


def save_and_predict(genotype, arch_id, args):
    genotype_save_foldername = os.path.join(args.working_directory,
                                            'history')

    if not os.path.exists(genotype_save_foldername):
        os.makedirs(genotype_save_foldername, exist_ok=False)

    with open(os.path.join(genotype_save_foldername,
                           'arch_%d.txt'%arch_id), 'w') as f:
        to_save = genotype if type(genotype) is str else str(genotype)
        f.write('%s'%(to_save))

    worker = Worker(working_directory=args.working_directory,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    run_id=arch_id,
                    scheme=args.scheme,
                    dataset=args.dataset,
                    nb201=True,
                    debug=args.debug)

    worker.compute(genotype,
                   budget=args.num_epochs,
                   config_id=(arch_id, 0, 0))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_directory', type=str,
                        help='directory where the generated results are saved')
    parser.add_argument('--num_iterations', type=int,
                        help='number of function evaluations performed.', default=400)
    parser.add_argument('--global_seed', type=int, default=1,
                        help='global seed number')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs to train the baselearner')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Mini-batch size to train the baselearner')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='image dataset')
    parser.add_argument('--population_size', type=int, default=50,
                        help='The population size to consider.')
    parser.add_argument('--sample_size', type=int, default=10,
                        help='The ensemble size to consider.')
    parser.add_argument('--severity_list', type=str, default='0',
                        help='Severity levels to sample from during evolution')
    parser.add_argument('--esa', type=str, default='beam_search',
                        help='Ensemble selection algorithm')
    parser.add_argument('--scheme', type=str, default='nes_re',
                        help='scheme name, i.e. nes or deepens variants')
    parser.add_argument('--nb201', action='store_true', default=False,
                        help='NAS-bench-201 space')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode: run for a single mini-batch')
    args = parser.parse_args()

    assert args.global_seed > 0, "global seed should be greater than 0"

    # set seed relative to the global seed and fix working directory
    args.working_directory = os.path.join(args.working_directory,
                                          'run_%d'%args.global_seed)
    if not os.path.exists(args.working_directory):
        os.makedirs(args.working_directory, exist_ok=True)

    with open('nes/utils/nb201/configs/arch_to_id.pkl', 'rb') as f:
        arch_to_id_dict = pickle.load(f)

    with open(os.path.join(args.working_directory, 'settings.txt'), 'w') as f:
        f.write(str(args))


    population = [] # this will be a list of tuples (string encodings, list_of_ops, ids)
    history = []
    arch_id = 0

    # sample initial random population
    for i in range(args.population_size):
        seed_id = (args.global_seed - 1) * BUDGET + arch_id
        np.random.seed(seed_id)
        random.seed(seed_id)

        arch_str, ops = sample_random_genotype()
        print(f'>>>>>>>> ITERATION {i} <<<<<<<<')
        print(f'+ arch_id: {arch_id}, arch_str: {arch_str}')

        # bookkeeping
        population.append((arch_str, ops, arch_id))
        history.append(arch_str)

        # get arch id from string
        nb201_id = str(arch_to_id_dict[arch_str][0])
        genotype = get_genotype(nb201_id)

        save_and_predict(genotype, arch_id, args)
        arch_id += 1


    for k in range(args.population_size, args.num_iterations):
        seed_id = (args.global_seed - 1) * BUDGET + arch_id
        np.random.seed(seed_id)
        random.seed(seed_id)

        print(f'>>>>>>>> ITERATION {k} <<<<<<<<')

        parent = select_new_parent(population, args.sample_size, args.scheme,
                                   args.working_directory, args.esa,
                                   args.severity_list)
        print(f'+ arch_id: {arch_id}, parent_arch_str: {parent[0]}')
        child_arch_ops = mutate_arch(parent)
        child_arch_str = get_arch_str(child_arch_ops)
        print(f'+ arch_id: {arch_id}, child_arch_str: {child_arch_str}')

        # bookkeeping
        population.append((child_arch_str, child_arch_ops, arch_id))
        history.append((child_arch_str, child_arch_ops, arch_id))

        # get nb201 id from string
        nb201_id = str(arch_to_id_dict[child_arch_str][0])
        genotype = get_genotype(nb201_id)

        save_and_predict(genotype, arch_id, args)
        arch_id += 1

        if len(population) >= args.population_size:
            population.pop(0)



