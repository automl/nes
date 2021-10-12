import os
import logging
import torch
from pathlib import Path

from nes.optimizers.baselearner_train.model import DARTSByGenotype as model_cifar
from nes.optimizers.baselearner_train.model_imagenet import DARTSByGenotype as model_tiny
from nes.optimizers.baselearner_train.utils import build_dataloader_by_sample_idx as build_dataloader
from nes.utils.cifar10_C_loader import build_dataloader_tiny


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

    if dataset == tiny:
        model_type = model_tiny
        dataloader_train = build_dataloader_tiny(
            batch_size=bslrn_batch_size, severity=0, mode='train',
            n_workers=n_workers)
        dataloader_val = build_dataloader_tiny(
            batch_size=bslrn_batch_size, severity=0, mode='val',
            n_workers=n_workers)
    else:
        model_type = model_cifar
        end_idx = 50000 if dataset == 'fmnist' else 40000

        if n_datapoints is not None: end_idx = n_datapoints
        dataloader_train = build_dataloader(data_path, bslrn_batch_size, mode,
                                            dataset, (0, end_idx), device,
                                            put_data_on_gpu=True)
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

