import os
import argparse
import logging
import torch
from pathlib import Path

from nes.optimizers.baselearner_train.model import DARTSByGenotype
from nes.optimizers.baselearner_train.utils import build_dataloader_by_sample_idx as build_dataloader
from nes.optimizers.baselearner_train.genotypes import Genotype, DARTS


def run_train(seed, arch_id, arch, num_epochs, bslrn_batch_size, exp_name,
              logger, data_path='data', mode='train', debug=False,
              dataset='fmnist'):
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

    Returns:
        None
    """
    device = torch.device(f'cuda:0')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    Path(exp_name).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(os.path.join(exp_name,
                                          f"arch{arch_id}.log"), mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    genotype = eval(arch)

    end_idx = 50000 if dataset == 'fmnist' else 40000
    dataloader = build_dataloader(
        data_path, bslrn_batch_size,
        mode, dataset, (0, end_idx), device,
        put_data_on_gpu=True
    )

    logger.info(f"[{mode}] (arch {arch_id}: {genotype}, init: {seed})...")

    DARTSByGenotype.base_learner_train_save(seed_init=seed,
                                            arch_id=arch_id,
                                            genotype=genotype,
                                            train_loader=dataloader,
                                            num_epochs=num_epochs,
                                            save_path=exp_name,
                                            device=device,
                                            dataset=dataset,
                                            verbose=True,
                                            debug=debug,
                                            logger=logger)

