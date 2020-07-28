import random
import torch
from collections import Counter

from nes.ensemble_selection.utils import evaluate_predictions, form_ensemble_pred, Registry


def run_esa(M, population, esa, val_severity):
    model_ids_pool = list(population.keys())
    models_preds_pool = {
        x: population[x].preds['val'][str(val_severity)] for x in model_ids_pool
    }
    model_ids_to_ensemble = esa(models_preds_pool, 'loss', M)['models_chosen']
    return model_ids_to_ensemble


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


