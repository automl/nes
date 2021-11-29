import os
import random
import warnings
import numpy as np
import pickle
warnings.filterwarnings('ignore')

from nes.ensemble_selection.config import BUDGET
from nes.nasbench201.re.re_sampler import (
    get_genotype,
    get_arch_str,
    select_new_parent,
    mutate_arch,
    sample_random_genotype,
    save_and_predict
)


def re_master(args):
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
                                   args.working_directory, "beam_search",
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_directory', type=str,
                        help='directory where the generated results are saved')
    parser.add_argument('--num_iterations', type=int,
                        help='number of function evaluations performed.', default=400)
    parser.add_argument('--global_seed', type=int, default=1,
                        help='global seed number')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='image dataset')
    parser.add_argument('--population_size', type=int, default=50,
                        help='The population size to consider.')
    parser.add_argument('--sample_size', type=int, default=10,
                        help='The ensemble size to consider.')
    parser.add_argument('--severity_list', type=str, default='0 5',
                        help='Severity levels to sample from during evolution')
    parser.add_argument('--scheme', type=str, default='nes_re',
                        help='scheme name, i.e. nes or deepens variants')
    parser.add_argument('--n_datapoints', type=int, default=40000,
                        help='train size')
    args = parser.parse_args()

    re_master(args)

