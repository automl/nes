import os
import re
import torch
import numpy as np
import ConfigSpace

from functools import partial, wraps
from pathlib import Path
from torch.autograd import Variable
from ConfigSpace.read_and_write import json as cs_json

from nes.darts.baselearner_train.genotypes import Genotype, PRIMITIVES


only_numeric_fn = lambda x: int(re.sub("[^0-9]", "", x))
custom_sorted = partial(sorted, key=only_numeric_fn)


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

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


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


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

