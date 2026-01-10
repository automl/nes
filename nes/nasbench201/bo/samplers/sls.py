import numpy as np

from nes.nasbench201.bo.utils import (
    evaluate_ensemble,
    OPS
)


def sls(args, eval_pool, arch_pool, ens_size, iterations, ls_steps,
        seed_archs_ids=None, seed_prob=.0):
    return_x = []
    return_y = []
    for _ in range(iterations):
        if seed_archs_ids is not None:
            choices_idx = np.random.choice(len(eval_pool)-seed_archs_ids, ens_size, replace=False)
            # replace one bsl with one of the new added ones to the eval_pool
            if sum(np.random.binomial(1, seed_prob, 1) == 1):
                list_of_seed_ids = list(range(len(eval_pool)))[-seed_archs_ids:]
                seed_id = np.random.choice(list_of_seed_ids)
                to_replace_id = np.random.choice(len(choices_idx))
                choices_idx[to_replace_id] = seed_id
        else:
            choices_idx = np.random.choice(len(eval_pool), ens_size, replace=False)

        ens_to_evaluate = [eval_pool[i] for i in choices_idx]
        x, y = evaluate_ensemble(ens_to_evaluate, args.working_directory,
                                 args.scheme, args.severity_list)
        x = [list(bslr[1]) for bslr in x]
        X = []
        for bslr in x:
            for op in bslr:
                X.append(OPS.index(op))
        best_x = X
        best_y = y
        # run ls_steps of local search
        for ls_iter in range(ls_steps):
            # list of all arch ids which are not in the current best ensemble
            ids = list(set(np.arange(len(eval_pool))) - set(choices_idx))
            # sample a new arch which is not in choices_idx
            new_sampled_arch = np.random.choice(ids)
            # replace a random arch in the choices_idx with the new one
            random_id = np.random.choice(len(choices_idx))
            new_choices_idx = np.copy(choices_idx)
            new_choices_idx[random_id] = new_sampled_arch
            ens_to_evaluate = [eval_pool[i] for i in new_choices_idx]
            x, y = evaluate_ensemble(ens_to_evaluate, args.working_directory,
                                     args.scheme, args.severity_list)
            x = [list(bslr[1]) for bslr in x]
            X = []
            for bslr in x:
                for op in bslr:
                    X.append(OPS.index(op))

            if y < best_y:
                best_x = X
                best_y = y
                choices_idx = new_choices_idx

        return_x.append(best_x)
        return_y.append(best_y)
    return return_x, return_y

