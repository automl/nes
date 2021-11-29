import os
import random
import warnings
import numpy as np
from random import choice
import pickle
from copy import deepcopy
warnings.filterwarnings('ignore')

from nes.optimizers.baselearner_train.utils import sample_random_genotype
from nes.optimizers.baselearner_train.genotypes import PRIMITIVES, Genotype
from nes.ensemble_selection.config import BUDGET

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_directory', type=str,
                        help='directory where the generated results are saved')
    parser.add_argument('--arch_id', type=int, default=0,
                        help='architecture id number')
    parser.add_argument('--global_seed', type=int, default=1,
                        help='global seed number')
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


    # apply mutations

    def mutate(genotype):
        cell_types = ['reduce', 'normal']
        cell_type = choice(cell_types)
        no_mutate_cell = list(filter(lambda i: i != cell_type, cell_types))[0]
        mutation_type = choice(['op', 'state', 'identity'])
        print(cell_type, mutation_type)

        to_mutate = eval('genotype.%s'%cell_type).copy()
        no_mutate = eval('genotype.%s'%no_mutate_cell).copy()
        concat = genotype.normal_concat

        if mutation_type == 'identity':
            new_genotype = deepcopy(genotype)
        elif mutation_type == 'op':
            new_primitives = [x for x in PRIMITIVES if x != 'none']
            op = choice(new_primitives)
            # sample one random edge
            edge_id = choice(list(range(len(to_mutate))))
            edge = to_mutate[edge_id]
            new_edge = (op, edge[1])
            # replace the op
            to_mutate[edge_id] = new_edge
        elif mutation_type == 'state':
            node_list = list(range(1,4))
            # sample random node
            node = choice(node_list)
            list_of_parent_nodes = [
                [], [0,1], [0,1,2], [0,1,2,3]
            ]
            edges = to_mutate[2*node: 2*node+2]
            edge_id = choice(range(2))
            edge = edges[edge_id]
            old_node_id = edge[1]

            parent_ids = list_of_parent_nodes[node]
            if old_node_id not in parent_ids:
                new_node_id = choice(parent_ids)
            else:
                new_node_id = choice([x for x in parent_ids if x != old_node_id])

            new_edge = (edge[0], new_node_id)
            to_mutate[2*node+edge_id] = new_edge

        if cell_type == 'normal':
            new_genotype = Genotype(
                normal=to_mutate, normal_concat=concat,
                reduce=no_mutate, reduce_concat=concat
            )
        else:
            new_genotype = Genotype(
                normal=no_mutate, normal_concat=concat,
                reduce=to_mutate, reduce_concat=concat
            )
        return new_genotype

    for i in range(1, 21):
        new_genotype = mutate(genotype)
        new_id = '%d%d'%(args.arch_id, i)

        with open(os.path.join(genotype_save_foldername,
                               'arch_%s.txt'%new_id), 'w') as f:
            to_save = new_genotype if type(new_genotype) is str else str(new_genotype)
            f.write('%s'%(to_save))

