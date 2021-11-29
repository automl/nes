import os
import logging
import torch
from pathlib import Path

from nes.darts.baselearner_train.model import DARTSByGenotype as model_cifar
from nes.darts.baselearner_train.model_imagenet import DARTSByGenotype as model_tiny
from nes.utils.data_loaders import build_dataloader_tiny as dataloader_tiny, \
                                   build_dataloader_fmnist as dataloader_fmnist, \
                                   build_dataloader_cifar_c as dataloader_cifar


def run_train(seed, arch_id, arch, num_epochs, bslrn_batch_size, exp_name,
              logger, data_path='data', mode='train', debug=False,
              anchor=False, dataset='fmnist', global_seed=0, n_workers=4,
              anch_coeff=1, n_datapoints=None, **kwargs):
    """Function that trains a given architecture.

    Args:
        seed                 (int): seed number
        arch_id              (int): architecture id
        arch                 (str): architecture genotype as string
        num_epochs           (int): number of epochs to train
        bslrn_batch_size     (int): mini-batch size
        exp_name             (str): directory where to save results
        logger    (logging.Logger): logger object
        data_path            (str): directory where the dataset is stored
        mode                 (str): train or validation
        debug               (bool): train for a single mini-batch only
        dataset              (str): dataset name
        global_seed          (int): global seed for optimizer runs
        n_workers            (int): number of workers for dataloaders
        anch_coeff           (int): coefficient for anchored ensembles
        n_datapoints         (int): number of total training data used

    Returns:
        None
    """
    device = torch.device(f'cuda:0')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    Path(exp_name).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(os.path.join(exp_name,
                                          f"arch{arch_id}.log"), mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    genotype = eval(arch)

    if dataset == 'tiny':
        model_type = model_tiny
        dataloader = dataloader_tiny
    else:
        model_type = model_cifar
        end_idx = 50000 if dataset == 'fmnist' else 40000
        dataloader = dataloader_fmnist if dataset == 'fmnist' else dataloader_cifar
        if n_datapoints is not None: end_idx = n_datapoints

    dataloader_train = dataloader(
        bslrn_batch_size, train=True, device=device,
        sample_indices=list(range(0, end_idx)), mode='train',
        dataset=dataset, n_workers=n_workers)
    if debug:
        end_idx_val = 60000 if dataset == 'fmnist' else 50000
        dataloader_val = dataloader(
            bslrn_batch_size, train=True, device=device, mode='val',
            sample_indices=list(range(end_idx, end_idx_val)),
            dataset=dataset, n_workers=n_workers)
    else:
        dataloader_val = None

    logger.info(f"[{mode}] (arch {arch_id}: {genotype}, init: {seed})...")

    model_type.base_learner_train_save(seed_init=seed,
                                       arch_id=arch_id,
                                       genotype=genotype,
                                       train_loader=dataloader_train,
                                       test_loader=dataloader_val,
                                       num_epochs=num_epochs,
                                       save_path=exp_name,
                                       device=device,
                                       dataset=dataset,
                                       verbose=True,
                                       debug=debug,
                                       global_seed=global_seed,
                                       anchor=anchor,
                                       anch_coeff=anch_coeff,
                                       logger=logger,
                                       **kwargs)

