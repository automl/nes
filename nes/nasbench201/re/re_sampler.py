import os
import copy
import random
import numpy as np

from nes.ensemble_selection.config import model_seeds
from nes.ensemble_selection.esas import registry as esas_registry
from nes.ensemble_selection.esas import run_esa
from nes.ensemble_selection.containers import load_baselearner
from nes.nasbench201 import NB201Worker as Worker


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

    parent_id = random.choice(pop_sample['models_chosen'])
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
                    run_id=arch_id,
                    scheme=args.scheme,
                    n_datapoints=args.n_datapoints,
                    dataset=args.dataset)

    worker.compute(config=genotype,
                   budget=200,
                   config_id=arch_id,
                   seed_id=arch_id)

