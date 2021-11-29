import torch
import numpy as np

from nes.ensemble_selection.containers import Baselearner
from nes.ensemble_selection.config import model_seeds
from nes.ensemble_selection.utils import get_dataloaders_and_classes
from nes.darts.baselearner_train.model import DARTSByGenotype as model_cifar
from nes.darts.baselearner_train.model_imagenet import DARTSByGenotype as model_tiny

# oneshot NAS
from nes.darts.baselearner_train.oneshot.darts_wrapper_discrete import DartsWrapper

# nb201 specific
from nes.utils.nb201.models import get_cell_based_tiny_net, CellStructure
from nes.utils.nb201.config_utils import dict2config
from nes.utils.nb201.api_utils import ResultsCount


def load_nn_module(state_dict_dir, genotype, init_seed=0, dataset='fmnist',
                   oneshot=False, nb201=False, **kwargs):
    # init_seed only used then genotype is None, i.e. when querying nasbench201

    if not nb201:
        # seed_init can be anything because we will load a state_dict 
        # anyway, so initialization doesn't matter.
        if not oneshot:
            model_type = model_cifar if dataset != "tiny" else model_tiny

            model = model_type(genotype=genotype, seed_init=0,
                               dataset=dataset, **kwargs)
            model.load_state_dict(torch.load(state_dict_dir))

            print("PARAMS: {}".format(
                np.sum(np.prod(v.size()) for name, v in model.named_parameters()
                      if "auxiliary" not in name)/1e6)
            )
        else:
            model_wrap = DartsWrapper(save_path=state_dict_dir, seed=init_seed,
                                      batch_size=100, grad_clip=0.25, epochs=100)
            model_wrap.set_model_weights_from_genotype(genotype)
            model = model_wrap.model

    else:
        # extract the predictions from the nasbench201 checkpoints
        dataset_to_nb201_dict = {
            'cifar10': 'cifar10-valid',
            'cifar100': 'cifar100',
            'imagenet': 'ImageNet16-120',
        }
        assert dataset in dataset_to_nb201_dict.keys()

        seed_list = [777, 888, 999]
        if init_seed == 0: seed = 777
        elif init_seed == 1: seed = 888
        else: seed = 999

        xdata = torch.load(state_dict_dir)

        try:
            odata = xdata['full']['all_results'][(dataset_to_nb201_dict[dataset],
                                                  seed)]
        except KeyError:
            seed_list.remove(seed)
            seed = seed_list[0]
            try:
                odata = xdata['full']['all_results'][(dataset_to_nb201_dict[dataset],
                                                      seed)]
            except KeyError:
                seed = seed_list[1]
                odata = xdata['full']['all_results'][(dataset_to_nb201_dict[dataset],
                                                      seed)]

        # load the saved model weights from the official NB201 dataset
        result = ResultsCount.create_from_state_dict(odata)
        result.get_net_param()
        arch_config = result.get_config(CellStructure.str2structure)
        net_config = dict2config(arch_config, None)
        model = get_cell_based_tiny_net(net_config)
        model.load_state_dict(result.get_net_param())

    return model


def create_baselearner(state_dict_dir, genotype, arch_seed, init_seed, scheme,
                       dataset, device, save_dir, oneshot=False, nb201=False,
                       n_datapoints=None, **kwargs):
    """
    A function which wraps an nn.Module with the Baselearner container, computes
    predictions and evaluations and finally saves everything.
    """
    assert dataset in ["cifar10", "cifar100", "fmnist", "imagenet", "tiny"]

    model_nn = load_nn_module(state_dict_dir, genotype, init_seed,
                              dataset=dataset, oneshot=oneshot, nb201=nb201,
                              **kwargs)

    severities = range(6) if (dataset in ["cifar10", "cifar100", "tiny"]) else range(1)

    model_id = model_seeds(arch_seed, init_seed, scheme)
    baselearner = Baselearner(model_id=model_id, severities=severities,
                              device=device, nn_module=model_nn)

    # Load dataloaders (val, test, all severities) to make predictions on
    dataloaders, num_classes = get_dataloaders_and_classes(dataset, device,
                                                           nb201, n_datapoints)

    baselearner.to_device(device)
    baselearner.compute_preds(dataloaders, severities, num_classes=num_classes)
    baselearner.compute_evals(severities)

    # saves the model_id, nn_module and preds & evals.
    baselearner.save(save_dir)
    return baselearner

