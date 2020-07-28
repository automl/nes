import torch

from nes.ensemble_selection.containers import Baselearner
from nes.ensemble_selection.utils import model_seeds, create_dataloader_dict_c10, create_dataloader_dict_fmnist
from nes.optimizers.baselearner_train.train import DARTSByGenotype


def load_nn_module(state_dict_dir, genotype, dataset='fmnist'):

    # seed_init can be anything because we will load a state_dict 
    # anyway, so initialization doesn't matter.
    model = DARTSByGenotype(genotype=genotype, seed_init=0,
                            dataset=dataset)
    model.load_state_dict(torch.load(state_dict_dir))

    return model


def create_baselearner(state_dict_dir, genotype, arch_seed, init_seed, scheme,
                       dataset, device, save_dir, severities):
    """
    A function which wraps an nn.Module with the Baselearner container, computes
    predictions and evaluations and finally saves everything.    
    """
    assert dataset in ["cifar10", "fmnist"]

    model_nn = load_nn_module(state_dict_dir, genotype, dataset=dataset)

    severities = range(6) if (dataset == "cifar10") else range(1)

    model_id = model_seeds(arch_seed, init_seed, scheme)
    baselearner = Baselearner(model_id=model_id, severities=severities, device=torch.device('cpu'),
                              nn_module=model_nn)

    # Load dataloaders (val, test, all severities) to make predictions on
    if dataset == 'fmnist':
        create_dataloader_dict = create_dataloader_dict_fmnist
    elif dataset == 'cifar10':
        create_dataloader_dict = create_dataloader_dict_c10

    dataloaders = create_dataloader_dict(device)

    baselearner.to_device(device)
    baselearner.compute_preds(dataloaders, severities)
    baselearner.compute_evals(severities)

    # saves the model_id, nn_module and preds & evals.
    baselearner.save(save_dir)

    return baselearner