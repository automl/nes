##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os
import sys
import hashlib
import pickle
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image


def calculate_md5(fpath, chunk_size=1024 * 1024):
  md5 = hashlib.md5()
  with open(fpath, 'rb') as f:
    for chunk in iter(lambda: f.read(chunk_size), b''):
      md5.update(chunk)
  return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
  return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
  if not os.path.isfile(fpath): return False
  if md5 is None: return True
  else          : return check_md5(fpath, md5)


class ImageNet16(data.Dataset):
  # http://image-net.org/download-images
  # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
  # https://arxiv.org/pdf/1707.08819.pdf

  # we only need the validation data
  valid_list = [
        ['val_data', '3410e3017fdaefba8d5073aaa65e4bd6'],
    ]

  def __init__(self, root, train, transform, use_num_of_class_only=None):
    self.root      = root
    self.transform = transform
    self.train     = train  # training set or valid set
    if not self._check_integrity(): raise RuntimeError('Dataset not found or corrupted.')

    downloaded_list = self.valid_list
    self.data    = []
    self.targets = []

    # now load the picked numpy arrays
    for i, (file_name, checksum) in enumerate(downloaded_list):
      file_path = os.path.join(self.root, file_name)
      with open(file_path, 'rb') as f:
        if sys.version_info[0] == 2:
          entry = pickle.load(f)
        else:
          entry = pickle.load(f, encoding='latin1')
        self.data.append(entry['data'])
        self.targets.extend(entry['labels'])
    self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)
    self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    if use_num_of_class_only is not None:
      assert isinstance(use_num_of_class_only, int) and use_num_of_class_only > 0 and use_num_of_class_only < 1000, 'invalid use_num_of_class_only : {:}'.format(use_num_of_class_only)
      new_data, new_targets = [], []
      for I, L in zip(self.data, self.targets):
        if 1 <= L <= use_num_of_class_only:
          new_data.append(I)
          new_targets.append(L)
      self.data    = new_data
      self.targets = new_targets

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index] - 1

    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    return img, target

  def __len__(self):
    return len(self.data)

  def _check_integrity(self):
    root = self.root
    for fentry in self.valid_list:
      filename, md5 = fentry[0], fentry[1]
      fpath = os.path.join(root, filename)
      if not check_integrity(fpath, md5):
        return False
    return True


