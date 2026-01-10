import os
import random
import warnings
import numpy as np
import pickle
warnings.filterwarnings('ignore')

from nes.ensemble_selection.config import BUDGET
from nes.nasbench201.bo.utils import (
    get_genotype,
    get_arch_str,
    select_baselearners,
    evaluate_ensemble,
    sample_random_genotype,
    save_and_predict,
    OPS
)

from nes.nasbench201.bo.surrogate_models.bananas_mlp import MetaNeuralnet
import tensorflow as tf
from nes.nasbench201.bo.acquisition_functions import acq_fn

from itertools import combinations


def create_combinations(ensembles, baselearner, ens_size):
    """
        ensembles: ndarray of size (N, sample_size*6)
        baselearner: ndarray of of size 6
    """
    candidates = []
    for config in ensembles:
        ens = np.array_split(config, ens_size)
        for combination in combinations(ens, ens_size-1):
            candidate_ensemble = list(combination) + [np.array(baselearner)]
            candidate_ensemble = [bslr.tolist() for bslr in candidate_ensemble]
            X = []
            for bslr in candidate_ensemble:
                X.extend(bslr)
            candidates.append(X)

    return candidates


def bo_master(args):
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


    architecture_pool = [] # list of tuples (string encodings, list_of_ops, ids)
    evaluation_pool = []
    query_data = {'x': [], 'y': []}
    arch_id = 0

    # sample initial random population
    for i in range(args.init_samples):
        seed_id = (args.global_seed - 1) * BUDGET + arch_id
        np.random.seed(seed_id)
        random.seed(seed_id)

        arch_str, ops = sample_random_genotype()
        print(f'>>>>>>>> ITERATION {i} <<<<<<<<')
        print(f'+ arch_id: {arch_id}, arch_str: {arch_str}')

        # bookkeeping
        evaluation_pool.append((arch_str, ops, arch_id))
        architecture_pool.append(ops)

        # get arch id from string
        nb201_id = str(arch_to_id_dict[arch_str][0])
        genotype = get_genotype(nb201_id)

        save_and_predict(genotype, arch_id, args)
        arch_id += 1

        #TODO: fix this later to >=
        if len(evaluation_pool) > args.sample_size:
            x, y = select_baselearners(evaluation_pool, args.sample_size, args.scheme,
                                       args.working_directory, "beam_search",
                                       args.severity_list)
            # every entry in x is a list of op str representing the baselearner
            x = [list(bslr[1]) for bslr in x]
            #TODO: change this representation later. For now it is a list of
            # ints with size sample_size * 6
            X = []
            for bslr in x:
                for op in bslr:
                    X.append(OPS.index(op))
            query_data['x'].append(X)
            query_data['y'].append(y)

    while len(evaluation_pool) < args.num_iterations:
        seed_id = (args.global_seed - 1) * BUDGET + arch_id
        np.random.seed(seed_id)
        random.seed(seed_id)

        xtrain = np.array(query_data['x'])
        ytrain = np.array(query_data['y']).reshape(-1, 1)

        # sample B candidate architectures
        candidate_archs = []
        ens_combinations = []
        for i in range(args.B):
            arch_str, ops = sample_random_genotype()
            xcand = [OPS.index(op) for op in ops]
            candidate_archs.append(xcand)
            ens_combinations.extend(create_combinations(xtrain, xcand,
                                                        args.sample_size))

        candidate_preds = []
        train_error = 0
        for _ in range(args.surrogate_ensemble_size):
            surr = MetaNeuralnet()
            train_error += surr.fit(xtrain, ytrain)

            candidate_preds.append(np.squeeze(surr.predict(np.array(ens_combinations))))

            # clear the tf graph. Used for the bananas surrogate
            tf.reset_default_graph()

        tf.keras.backend.clear_session()

        train_error /= args.surrogate_ensemble_size
        print('Meta neural net train error: {}'.format(train_error))

        # compute the acquisition function for all the candidate architectures
        candidate_indices = acq_fn(candidate_preds, 'its')[:args.bo_batch_size]

        ensembles_selected = np.array(ens_combinations)[candidate_indices]
        query_data['x'] += ensembles_selected.tolist()

        # 6 is the number of ops in every baselearner. We want to train only
        # the last baselearner since the other ones are already trained and in
        # the evaluation pool.
        baselearners_to_train = set(tuple(ens) for ens in
                                    ensembles_selected[:, -6:].tolist())

        if (len(evaluation_pool) + len(baselearners_to_train)) > args.num_iterations:
            baselearners_to_train = list(baselearners_to_train)[
                :(args.num_iterations - len(evaluation_pool))
            ]

        for bslr in baselearners_to_train:
            ops = [OPS[op] for op in bslr]
            arch_str = get_arch_str(ops)
            # bookkeeping
            evaluation_pool.append((arch_str, np.array(ops), arch_id))
            architecture_pool.append(ops)

            # get arch id from string
            nb201_id = str(arch_to_id_dict[arch_str][0])
            genotype = get_genotype(nb201_id)
            save_and_predict(genotype, arch_id, args)
            arch_id += 1

        for ens in ensembles_selected:
            ens_bslr = np.array_split(ens, args.sample_size)
            arch_strs = [get_arch_str([OPS[op] for op in bslr]) for bslr in
                                      ens_bslr]
            ens_to_evaluate = list(filter(lambda x: x[0] in arch_strs,
                                          evaluation_pool))
            _, y = evaluate_ensemble(ens_to_evaluate, args.working_directory,
                                     args.scheme, args.severity_list)
            query_data['y'].append(y)

        #x, y = select_baselearners(evaluation_pool, args.sample_size, args.scheme,
        #                           args.working_directory, "beam_search",
        #                           args.severity_list)
        print(f'>>>>>>>> Eval Pool Size {len(evaluation_pool)} <<<<<<<<')
        #from IPython import embed
        #embed()
        #break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_directory', type=str,
                        help='directory where the generated results are saved')
    parser.add_argument('--num_iterations', type=int,
                        help='number of function evaluations performed.', default=400)
    parser.add_argument('--global_seed', type=int, default=1,
                        help='global seed number')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='image dataset')
    parser.add_argument('--surrogate_ensemble_size', type=int, default=3,
                        help='number of baselearners in surrogate ensemble.')
    parser.add_argument('--init_samples', type=int, default=50,
                        help='Initial random configurations.')
    parser.add_argument('--sample_size', type=int, default=3,
                        help='The ensemble size to consider.')
    parser.add_argument('--B', type=int, default=10,
                        help='Number of random configs sampled during each BO loop.')
    parser.add_argument('--bo_batch_size', type=int, default=10,
                        help='Number of configurations to evaluate during each BO loop.')
    parser.add_argument('--severity_list', type=str, default='0 5',
                        help='Severity levels to sample from during optimization')
    parser.add_argument('--scheme', type=str, default='nes_bo',
                        help='scheme name, i.e. nes or deepens variants')
    parser.add_argument('--n_datapoints', type=int, default=40000,
                        help='train size')
    args = parser.parse_args()

    bo_master(args)

