import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
import ConfigSpace

from functools import partial, wraps
from pathlib import Path
from torch.utils.data import DataLoader, SubsetRandomSampler
from ConfigSpace.read_and_write import json as cs_json

from nes.optimizers.baselearner_train.genotypes import Genotype, PRIMITIVES


only_numeric_fn = lambda x: int(re.sub("[^0-9]", "", x))
custom_sorted = partial(sorted, key=only_numeric_fn)


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


def build_dataloader_by_sample_idx(data_path, batch_size, mode,
                                   dataset, training_idxs, device,
                                   put_data_on_gpu=True):
    start_index = training_idxs[0]
    end_index = training_idxs[1]

    # transforms are scale + center
    transformations = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5,
                                                                  0.5), (0.5,
                                                                         0.5,
                                                                         0.5))]
    if dataset == 'fmnist':
        transformations = [transforms.ToTensor(), transforms.ToPILImage(),
                           transforms.Grayscale(num_output_channels=3)] + transformations
        dataset_cls = torchvision.datasets.FashionMNIST
    elif dataset == 'cifar10':
        dataset_cls = torchvision.datasets.CIFAR10

    trans = transforms.Compose(transformations)

    dataset = dataset_cls(
        root=data_path,
        train=True if mode != 'test' else False,
        transform=trans,
        download=True
    )

    # Data loader
    if mode != 'test':
        mask = [False] * len(dataset)
        for i in range(start_index, end_index):
            mask[i] = True

    if put_data_on_gpu:
        print("Putting data on GPU...")

        loader_full = DataLoader(
            dataset=dataset,
            sampler=SubsetRandomSampler(np.where(mask)[0]) if mode != 'test' else None,
            batch_size=len(dataset),
            shuffle=False,
            pin_memory=False
        )

        torch.manual_seed(0)
        all_data = next(iter(loader_full))
        all_data_on_gpu = []
        for data in all_data:
            all_data_on_gpu.append(data.to(device))
        all_data = all_data_on_gpu

        dataset = torch.utils.data.TensorDataset(*all_data)

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )
    else:
        loader = DataLoader(
            dataset=dataset,
            sampler=SubsetRandomSampler(np.where(mask)[0]) if mode != 'test' else None,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

    return loader


def sample_random_genotype(steps, multiplier):
    """Function to sample a random genotype (architecture).

    Args:
        steps      (int): number of intermediate nodes in the DARTS cell
        multiplier (int): number of nodes to concatenate in the output cell

    Returns:
        nes.optimizers.baselearner_train.genotypes.Genotype:
            the randomly sampled genotype
    """
    def _parse():
        gene = []
        n = 2
        start = 0
        for i in range(steps):
            end = start + n
            edges = np.random.choice(range(i + 2), 2, False).tolist()

            for j in edges:
                k_best = np.random.choice(list(range(8)))
                while k_best == PRIMITIVES.index('none'):
                    k_best = np.random.choice(list(range(8)))
                gene.append((PRIMITIVES[k_best], j))
            start = end
            n += 1
        return gene

    gene_normal, gene_reduce = _parse(), _parse()
    concat = range(2+steps-multiplier, steps+2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


def create_genotype(func):
    @wraps(func)
    def genotype_wrapper(*args, **kwargs):
        normal = func(*args, cell_type='normal', **kwargs)
        reduction = func(*args, cell_type='reduce', **kwargs)
        concat = list(range(2, 6))
        return Genotype(normal, concat, reduction, concat)
    return genotype_wrapper


@create_genotype
def parse_config(config, config_space, cell_type):
    """Function that converts a ConfigSpace representation of the architecture
        to a Genotype.
    """
    cell = []
    config = ConfigSpace.Configuration(config_space, config)

    edges = custom_sorted(
        list(
            filter(
                re.compile('.*edge_{}*.'.format(cell_type)).match,
                config_space.get_active_hyperparameters(config)
            )
        )
    ).__iter__()

    nodes = custom_sorted(
        list(
            filter(
                re.compile('.*inputs_node_{}*.'.format(cell_type)).match,
                config_space.get_active_hyperparameters(config)
            )
        )
    ).__iter__()

    op_1 = config[next(edges)]
    op_2 = config[next(edges)]
    cell.extend([(op_1, 0), (op_2, 1)])

    for node in nodes:
        op_1 = config[next(edges)]
        op_2 = config[next(edges)]
        input_1, input_2 = map(int, config[node].split('_'))
        cell.extend([(op_1, input_1), (op_2, input_2)])

    return cell

