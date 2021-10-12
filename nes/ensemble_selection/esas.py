import random
import torch
import torch.nn as nn
import warnings
import numpy as np
from collections import Counter

from nes.ensemble_selection.utils import evaluate_predictions, form_ensemble_pred, form_ensemble_pred_v_2, Registry, classif_accuracy


def run_esa(M, population, esa, val_severity, validation_size=-1, diversity_strength=None):
    model_ids_pool = list(population.keys())
    models_preds_pool = {
        x: population[x].preds['val'][str(val_severity)] for x in model_ids_pool
    }

    if validation_size > -1:
        assert validation_size > 0, "Validation size cannot be 0."
        _models_preds_pool = {}
        for x, tensor_data in models_preds_pool.items():
            preds, labels = tensor_data.tensors
            assert (validation_size <= len(preds)), "Validation size too large."

            _tensor_data = torch.utils.data.TensorDataset(preds[:validation_size],
                                                          labels[:validation_size])
            _models_preds_pool[x] = _tensor_data
        models_preds_pool = _models_preds_pool

    if diversity_strength is not None:
        esa_out = esa(models_preds_pool, 'loss', M, div_strength=diversity_strength)
    else:
        esa_out = esa(models_preds_pool, 'loss', M)
    return esa_out
    # if 'weights' in esa_out.keys():
    #     model_ids_to_ensemble = esa_out['models_chosen']
    #     weights = esa_out['weights']
    #     return {'models_chosen':}
    # else:
    #     model_ids_to_ensemble = esa_out['models_chosen']
    #     return model_ids_to_ensemble


# ---------------------------------------------------------------------------- #
#                         Ensemble selection algorithms                        #
# ---------------------------------------------------------------------------- #

# Note, one can add other ensemble selection algorithms (ESA) here.
# Decorate them similar to the example below with a label, and 
# use the label wherever `--esa` appears in scripts found in 
# `cluster_scripts/`.

registry = Registry()

# Note "beam_search" is the *forward selection without replacement*.
@registry('beam_search')
def beam_search(models_preds, metric_to_use, M, beam_width=1, lsm_applied=False):
    assert M > 1

    if len(models_preds) <= M:
        return {"models_chosen": list(models_preds.keys())}

    assert metric_to_use in ["loss"]  # , "acc"]

    models_keys = list(models_preds.keys())
    labels = models_preds[models_keys[0]].tensors[1]

    # can probably simplify this code by combining the stuff below.
    models_evals = {}
    for key in models_keys:
        model_preds = models_preds[key]
        models_evals[key] = evaluate_predictions(model_preds, lsm_applied=lsm_applied)

    models_ranking = sorted(models_evals, key=lambda x: models_evals[x][metric_to_use])

    beams = [frozenset({models_ranking[x]}) for x in range(beam_width)]

    m = 1
    while m < M:
        beam_pool = set([beam.union({x}) for beam in beams for x in models_keys if x not in beam])

        losses = {}
        for beam in beam_pool:
            models_to_try = {net: models_preds[net].tensors[0] for net in beam}

            ens_predictions = form_ensemble_pred(models_to_try, lsm_applied=False)
            ens_preds = torch.utils.data.TensorDataset(ens_predictions, labels)

            ens_eval = evaluate_predictions(ens_preds, lsm_applied=True)

            losses[beam] = ens_eval[metric_to_use]

        beams_ranking = sorted(losses, key=losses.get)
        beams = beams_ranking[:beam_width]
        m += 1

    return {"models_chosen": list(beams_ranking[0])}


@registry('quick_and_greedy')
def quick_greed(models_preds, metric_to_use, M, lsm_applied=False):
    # assert len(models_preds) >= M
    assert M > 1

    if len(models_preds) <= M:
        return {"models_chosen": list(models_preds.keys())}

    assert metric_to_use in ["loss"]  # , "acc"]

    models_keys = list(models_preds.keys())
    labels = models_preds[models_keys[0]].tensors[1]

    models_evals = {}
    for key in models_keys:
        model_preds = models_preds[key]
        models_evals[key] = evaluate_predictions(model_preds, lsm_applied=lsm_applied)

    models_ranking = sorted(models_evals, key=lambda x: models_evals[x][metric_to_use])

    models_to_keep = [models_ranking[0]]
    best_loss = models_evals[models_ranking[0]][metric_to_use]
    num_skipped = 0
    for ranked in models_ranking[1:]:
        models_to_try = {net: models_preds[net].tensors[0] for net in models_to_keep + [ranked]}

        ens_predictions = form_ensemble_pred(models_to_try, lsm_applied=False)
        ens_preds = torch.utils.data.TensorDataset(ens_predictions, labels)

        ens_eval = evaluate_predictions(ens_preds, lsm_applied=True)

        if ens_eval[metric_to_use] < best_loss:
            models_to_keep = list(models_to_try.keys())
            best_loss = ens_eval[metric_to_use]
        else:
            num_skipped += 1

        if len(models_to_keep) == M:
            break

    if len(models_to_keep) < M:
        print(f"Ensemble returned has size {len(models_to_keep)}, which is less than {M}.")

    print(f'Skipped {num_skipped} ranked models during ensemble construction.')

    return {"models_chosen": models_to_keep} 


@registry('top_M')
def choose_top_members(models_preds, metric_to_use, M, lsm_applied=False):
    # assert len(models_preds) >= M
    if len(models_preds) <= M:
        return {"models_chosen": list(models_preds.keys())}
    assert M > 1

    # if len(models_preds) == M:
    #     return {"models_chosen": list(models_preds.keys())}  # , "eval": best_loss, "num_skipped": num_skipped}

    assert metric_to_use in ["loss"]  # , "acc"]

    # if metric_to_use == 'acc':
    #     raise NotImplementedError("check out the sorting...")

    models_keys = list(models_preds.keys())

    models_evals = {}
    for key in models_keys:
        model_preds = models_preds[key]
        models_evals[key] = evaluate_predictions(model_preds, lsm_applied=lsm_applied)

    models_ranking = sorted(models_evals, key=lambda x: models_evals[x][metric_to_use])

    models_to_keep = models_ranking[:M]

    return {"models_chosen": models_to_keep}


# ---------------------------------------------------------------------------- #
#                                    STACKER                                   #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
# from my_tools import *
from nes.ensemble_selection.utils import classif_accuracy
import numpy as np
import warnings


# def plot_alphas(alphas, save_dir):
#     fig, ax = plt.subplots()
#     for x in alphas.squeeze().transpose(1, 0):
#         ax.plot(x)

#     fig.savefig(save_dir)
#     plt.close()

class Stacker(nn.Module):
    def __init__(self, N=1):
        super(Stacker, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, N)) # N = number of members to stack

    def forward(self, input):
        out = self.alpha.softmax(1).matmul(input)
        out = torch.squeeze(out, dim=1).log()
        return out  # returns the log of the combined probability vector

def optimize_stacker_weights(preds_loader, device, max_iter, lambda1=0.05, tol=1e-3, lr=15, verbose=False, save_plot_alphas=False):
    '''
    Btw, I removed the first input of this function. And a line changed is
    stacker = Stacker(len(models))
    to
    stacker = Stacker(val_loader.dataset.tensors[0].shape[1])
    val_loader is supposed to be a loader for a TensorDataset where tensor[1] is the label tensor. tensor[0] should be of size
    (num_datapts, num_models, num_classes), where along dim 1 we stack the predictions of each model. The predictions
    are expected to be before softmax.
    '''
    # torch.manual_seed(1)
    stacker = Stacker(N=preds_loader.dataset.tensors[0].shape[1])
    stacker.to(device)

    # stack_ensemble = StackEnsemble(stacker)
    if len(preds_loader) != 1:
        raise ValueError("Batch size for stacker training must be full.")

    # Loss and optimizer
    nll = nn.NLLLoss()

    optimizer = torch.optim.SGD(stacker.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)

    # Train the stacker
    total_step = len(preds_loader)
    loss_list = []
    acc_list = []
    alphas = []

    # Save initialization of alpha
    alphas.append(stacker.alpha.softmax(1).detach().cpu().numpy())

    features, labels = next(iter(preds_loader))

    if features.requires_grad or labels.requires_grad:
        raise ValueError("preds_loader is giving stuff that requires gradient.")

    # Run the forward pass
    features = features.to(device)
    labels = labels.to(device)

    features = features.softmax(2)

    for iteration in range(max_iter):

        outputs = stacker(features)

        loss = nll(outputs, labels)
        loss_list.append(loss.item())
        reg_loss = (stacker.alpha.softmax(1) ** 2).sum()  # torch.norm(stacker.alpha.softmax(1) - unif_pvec, p=1)
        loss += -lambda1 * reg_loss

        # Backprop and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track alphas
        alphas.append(stacker.alpha.softmax(1).detach().cpu().numpy())

        # scheduler.step()

        # Track the accuracy and print progress
        if verbose:
            accuracy = classif_accuracy(outputs, labels)
            if (iteration + 1) % 1 == 0:
                print('Iter [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(iteration + 1, max_iter, loss.item(),
                                                                             (accuracy) * 100))

                # print(stacker.alpha.grad * lr)

        if (np.abs(alphas[-1] - alphas[-2]) < tol).all():
            accuracy = classif_accuracy(outputs, labels)

            print(f"Completed stacking weights optimization in {iteration + 1} steps.")
            print(f"Training NLL: {round(loss_list[-1], 3)}")
            print(f"Training accuracy: {round(accuracy * 100, 2)}%")

            break

    if iteration == max_iter - 1:
        print("Convergence criteria not met within the given max_iter.")

    # if save_plot_alphas:
    #     plot_alphas(np.array(alphas), "/home/zelaa/NIPS20/Sheh/robust_ensembles/plots/debug/alphas.pdf")

    return np.array(alphas), stacker.alpha.detach().cpu().numpy()



@registry("linear_unweighted_stack")
def choose_by_stacking(models_preds, metric_to_use, M, lsm_applied=False):
    '''
    assumes lsm not applied to models!
    '''
    if len(models_preds) <= M:
        return {"models_chosen": list(models_preds.keys())}
    assert M > 1

    # if len(models_preds) == M:
    #     return {"models_chosen": list(models_preds.keys())}  # , "eval": best_loss, "num_skipped": num_skipped}

    assert metric_to_use in ["loss"]  # , "acc"]

    models_keys = list(models_preds.keys())
    labels = models_preds[models_keys[0]].tensors[1]

    preds_tensor = torch.stack([models_preds[key].tensors[0] for key in models_keys], dim=1)

    preds_dst = torch.utils.data.TensorDataset(preds_tensor, labels)
    preds_loader = torch.utils.data.DataLoader(preds_dst, len(preds_dst), shuffle=False)

    device = preds_tensor.device

    lr = 2 * len(models_preds)

    alphas, alpha_psm = optimize_stacker_weights(preds_loader, device, 1000, lambda1=0, tol=1e-7, lr=lr, verbose=True, save_plot_alphas=False)

    # import ipdb; ipdb.set_trace()

    alpha_psm = alpha_psm.squeeze()

    models_to_keep = [models_keys[x] for x in np.argsort(-alpha_psm)[:M]]

    return {"models_chosen": models_to_keep}


@registry('forw_select_replace')
def forward_select(models_preds, metric_to_use, M, lsm_applied=False):
    # assert len(models_preds) >= M
    if len(models_preds) <= M:
        return {"models_chosen": list(models_preds.keys())}
    assert M > 1

    assert metric_to_use in ["loss"]  # , "acc"]

    models_keys = list(models_preds.keys())
    labels = models_preds[models_keys[0]].tensors[1]

    models_evals = {}
    for key in models_keys:
        model_preds = models_preds[key]
        models_evals[key] = evaluate_predictions(model_preds, lsm_applied=lsm_applied)

    models_ranking = sorted(models_evals, key=lambda x: models_evals[x][metric_to_use])

    models_kept = [models_ranking[0]]

    m = 1
    while m < M:
        losses = {}
        for candidate in models_ranking:
            models_to_try_preds = [models_preds[net].tensors[0] for net in (models_kept + [candidate])]

            ens_predictions = form_ensemble_pred_v_2(models_to_try_preds, lsm_applied=False)
            ens_preds = torch.utils.data.TensorDataset(ens_predictions, labels)

            ens_eval = evaluate_predictions(ens_preds, lsm_applied=True)

            losses[candidate] = ens_eval[metric_to_use]

        candidate_ranking = sorted(losses, key=losses.get)
        models_kept.append(candidate_ranking[0])
        m += 1

    print(f'Model frequencies: {Counter(models_kept).values()}')
    print(f'Num unique models: {len(Counter(models_kept).values())} \n')

    return {"models_chosen": models_kept}



@registry("linear_weighted_stack")
def choose_by_weighted_stacking(models_preds, metric_to_use, M, lsm_applied=False):
    '''
    assumes lsm not applied to models!
    '''
    if len(models_preds) <= M:
        return {"models_chosen": list(models_preds.keys())}
    assert M > 1

    # if len(models_preds) == M:
    #     return {"models_chosen": list(models_preds.keys())}  # , "eval": best_loss, "num_skipped": num_skipped}

    assert metric_to_use in ["loss"]  # , "acc"]

    models_keys = list(models_preds.keys())
    labels = models_preds[models_keys[0]].tensors[1]

    preds_tensor = torch.stack([models_preds[key].tensors[0] for key in models_keys], dim=1)

    preds_dst = torch.utils.data.TensorDataset(preds_tensor, labels)
    preds_loader = torch.utils.data.DataLoader(preds_dst, len(preds_dst), shuffle=False)

    device = preds_tensor.device

    lr = 2 * len(models_preds)

    alphas, alpha_psm = optimize_stacker_weights(preds_loader, device, 1000, lambda1=0, tol=1e-7, lr=lr, verbose=True, save_plot_alphas=False)

    alpha_sm = torch.tensor(alpha_psm).softmax(1).squeeze(0)
    topk, indices = torch.topk(alpha_sm, M)
    _sparse_alpha = torch.zeros_like(alpha_sm)
    sparse_alpha = _sparse_alpha.scatter(0, indices, topk)
    sparse_alpha = sparse_alpha / sparse_alpha.sum()

    models_to_keep = [models_keys[x] for x in indices]
    weights = np.array([sparse_alpha[x] for x in indices])

    return {"models_chosen": models_to_keep, "weights": weights}


@registry('beam_search_bma_acc')
def beam_search_bma_acc(models_preds, metric_to_use, M, beam_width=1, lsm_applied=False):
    assert M > 1

    if len(models_preds) <= M:
        return {"models_chosen": list(models_preds.keys())}

    assert metric_to_use in ["loss"]  # , "acc"]

    models_keys = list(models_preds.keys())
    labels = models_preds[models_keys[0]].tensors[1]

    # can probably simplify this code by combining the stuff below.
    models_evals = {}
    for key in models_keys:
        model_preds = models_preds[key]
        models_evals[key] = evaluate_predictions(model_preds, lsm_applied=lsm_applied)

    models_ranking = sorted(models_evals, key=lambda x: models_evals[x][metric_to_use])

    beams = [frozenset({models_ranking[x]}) for x in range(beam_width)]

    m = 1
    while m < M:
        beam_pool = set([beam.union({x}) for beam in beams for x in models_keys if x not in beam])

        losses = {}
        for beam in beam_pool:
            models_to_try = {net: models_preds[net].tensors[0] for net in beam}

            ens_predictions = form_ensemble_pred(models_to_try, lsm_applied=False)
            ens_preds = torch.utils.data.TensorDataset(ens_predictions, labels)

            ens_eval = evaluate_predictions(ens_preds, lsm_applied=True)

            losses[beam] = ens_eval[metric_to_use]

        beams_ranking = sorted(losses, key=losses.get)
        beams = beams_ranking[:beam_width]
        m += 1

    models_chosen = list(beams_ranking[0])
    weights = np.array([models_evals[k]['acc'] for k in models_chosen])
    weights = weights / weights.sum()

    return {"models_chosen": models_chosen, "weights": weights.astype("float32")}

@registry('beam_search_bma_loss')
def beam_search_bma_loss(models_preds, metric_to_use, M, beam_width=1, lsm_applied=False):
    assert M > 1

    if len(models_preds) <= M:
        return {"models_chosen": list(models_preds.keys())}

    assert metric_to_use in ["loss"]  # , "acc"]

    models_keys = list(models_preds.keys())
    labels = models_preds[models_keys[0]].tensors[1]

    # can probably simplify this code by combining the stuff below.
    models_evals = {}
    for key in models_keys:
        model_preds = models_preds[key]
        models_evals[key] = evaluate_predictions(model_preds, lsm_applied=lsm_applied)

    models_ranking = sorted(models_evals, key=lambda x: models_evals[x][metric_to_use])

    beams = [frozenset({models_ranking[x]}) for x in range(beam_width)]

    m = 1
    while m < M:
        beam_pool = set([beam.union({x}) for beam in beams for x in models_keys if x not in beam])

        losses = {}
        for beam in beam_pool:
            models_to_try = {net: models_preds[net].tensors[0] for net in beam}

            ens_predictions = form_ensemble_pred(models_to_try, lsm_applied=False)
            ens_preds = torch.utils.data.TensorDataset(ens_predictions, labels)

            ens_eval = evaluate_predictions(ens_preds, lsm_applied=True)

            losses[beam] = ens_eval[metric_to_use]

        beams_ranking = sorted(losses, key=losses.get)
        beams = beams_ranking[:beam_width]
        m += 1

    models_chosen = list(beams_ranking[0])
    _weights = np.array([models_evals[k]['loss'] for k in models_chosen]) # this is negative log likelihood
    weights_likelihoods = np.exp(-_weights)
    weights = weights_likelihoods / weights_likelihoods.sum()

    return {"models_chosen": models_chosen, "weights": weights.astype("float32")}



def compute_div(bsls_dct, ens_preds):
    assert all(b.shape == ens_preds.shape for b in bsls_dct.values())
    assert len(ens_preds.shape) == 2

    bsl_stack = torch.stack(list(bsls_dct.values()), dim=1).softmax(2)
    ens = ens_preds.unsqueeze(1).repeat(1, len(bsls_dct), 1).exp()

    # import ipdb; ipdb.set_trace()
    return (bsl_stack - ens).pow(2).sum(2).mean() 


# Note "beam_search" is the *forward selection without replacement*.
@registry('beam_search_with_div')
def beam_search_with_div(models_preds, metric_to_use, M, beam_width=1, lsm_applied=False, div_strength=1.0):
    assert M > 1

    if len(models_preds) <= M:
        return {"models_chosen": list(models_preds.keys())}

    assert metric_to_use in ["loss"]  # , "acc"]

    models_keys = list(models_preds.keys())
    labels = models_preds[models_keys[0]].tensors[1]

    # can probably simplify this code by combining the stuff below.
    models_evals = {}
    for key in models_keys:
        model_preds = models_preds[key]
        models_evals[key] = evaluate_predictions(model_preds, lsm_applied=lsm_applied)

    models_ranking = sorted(models_evals, key=lambda x: models_evals[x][metric_to_use])

    beams = [frozenset({models_ranking[x]}) for x in range(beam_width)]

    m = 1
    while m < M:
        beam_pool = set([beam.union({x}) for beam in beams for x in models_keys if x not in beam])

        losses = {}
        for beam in beam_pool:
            models_to_try = {net: models_preds[net].tensors[0] for net in beam}

            ens_predictions = form_ensemble_pred(models_to_try, lsm_applied=False)
            ens_preds = torch.utils.data.TensorDataset(ens_predictions, labels)
            ens_eval = evaluate_predictions(ens_preds, lsm_applied=True)

            diversity = compute_div(bsls_dct=models_to_try, ens_preds=ens_predictions)

            # print(ens_eval[metric_to_use], diversity)

            losses[beam] = ens_eval[metric_to_use] - div_strength * diversity

        beams_ranking = sorted(losses, key=losses.get)
        beams = beams_ranking[:beam_width]
        m += 1

    return {"models_chosen": list(beams_ranking[0])} 





