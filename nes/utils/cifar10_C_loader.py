import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from PIL import Image

from data.corruptions import *


DATA_PATH = "data"

d = {}
d['Gaussian Noise'] = gaussian_noise
d['Shot Noise'] = shot_noise
d['Impulse Noise'] = impulse_noise
d['Defocus Blur'] = defocus_blur
d['Glass Blur'] = glass_blur
d['Motion Blur'] = motion_blur
d['Zoom Blur'] = zoom_blur
d['Snow'] = snow
d['Frost'] = frost
d['Fog'] = fog
d['Brightness'] = brightness
d['Contrast'] = contrast
d['Elastic'] = elastic_transform
d['Pixelate'] = pixelate
d['JPEG'] = jpeg_compression

d['Speckle Noise'] = speckle_noise
d['Gaussian Blur'] = gaussian_blur
d['Spatter'] = spatter
d['Saturate'] = saturate


def corruption_choices(validation=None):
    if validation is None:
        return ['Gaussian Noise', 'Shot Noise', 'Impulse Noise',
                'Defocus Blur', 'Glass Blur', 'Motion Blur', 'Zoom Blur',
                'Snow', 'Frost', 'Fog', 'Brightness', 'Contrast', 'Elastic',
                'Pixelate', 'JPEG', 'Speckle Noise', 'Gaussian Blur',
                'Spatter', 'Saturate']
    elif validation:
        return ['Speckle Noise', 'Gaussian Blur', 'Spatter', 'Saturate']
    elif not validation:
        return ['Gaussian Noise', 'Shot Noise', 'Impulse Noise',
                'Defocus Blur', 'Glass Blur', 'Motion Blur', 'Zoom Blur',
                'Snow', 'Frost', 'Fog', 'Brightness', 'Contrast', 'Elastic',
                'Pixelate', 'JPEG']
    else:
        raise ValueError("Give a valid choice for whether "
                         "to return validation corruptions or not.")


# ============================
# Data loaders for CIFAR10-C
# ============================
class TensorDatasetWithTrans(Dataset):
    """
    TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def build_dataloader_cifar10_c(batch_size, corruption, severity, train, device,
                               sample_indices=None):
    '''
    Builds a CIFAR-10 dataloader of corrupted images. The corruption type and severity are inputs.
    :param batch_size: batch size of loader
    :param corruption: corruption as string. has to be one of the strings in the dictionary d above.
    :param severity: integer between 1 and 5 inclusive. 5 indicates high corruption
    :param train: train split or not (bool)
    :param device: device to load the data onto (maybe the device where the model is trained for speed?)
    :param sample_indices: range of indices of the datapoints to include (tuple of length 2). e.g. (20000, 30000) when
    Train = True, or (1000, 5000) is train = False.
    :return: data loader
    '''
    if not (0 <= severity <= 5):
        raise ValueError("Severity level must be in [1, 2, 3, 4, 5].")

    corruption_choices = ['Gaussian Noise', 'Shot Noise', 'Impulse Noise',
                          'Defocus Blur', 'Glass Blur', 'Motion Blur',
                          'Zoom Blur', 'Snow', 'Frost', 'Fog', 'Brightness',
                          'Contrast', 'Elastic', 'Pixelate', 'JPEG',
                          'Speckle Noise', 'Gaussian Blur', 'Spatter', 'Saturate']

    if not corruption in (corruption_choices + ["clean"]):
        raise ValueError(f"No corruption of type {corruption}.")

    if sample_indices is not None:
        start_index = sample_indices[0]
        end_index = sample_indices[1]
    else:
        start_index = 0
        if train:
            end_index = 50000
        else:
            end_index = 10000

    if corruption != "clean":
        images_path = DATA_PATH + '/cifar10-C/' + d[
            corruption].__name__ + f'/severity_{severity}'
        labels_path = DATA_PATH + '/cifar10-C/' + d[
            corruption].__name__ + f'/severity_{severity}'

        images = np.load(
            images_path + f'/images_{"train" if train else "test"}.npy'
        )[start_index:end_index]

        labels = np.load(
            labels_path + f'/labels_{"train" if train else "test"}.npy'
        )[start_index:end_index]
    else:
        dataset = dset.CIFAR10(DATA_PATH, train=train, download=True)
        images = dataset.data
        labels = np.array(dataset.targets)

        images = images[start_index:end_index]
        labels = labels[start_index:end_index]

    images = torch.stack([trn.ToTensor()(image) for image in images]).to(device)
    labels = torch.tensor(labels).to(device)

    # transforms are scale + center (with same mean, std as training data)
    trans = trn.Compose([trn.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = TensorDatasetWithTrans(tensors=(images, labels), transform=trans)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=False, pin_memory=False)

    return data_loader


def build_dataloader_cifar10_c_mixed(batch_size, validation_corruptions,
                                     severity, train, device,
                                     sample_indices=None):
    '''
    Builds a CIFAR-10 dataloader of corrupted images. The corruption for each datapoint is picked randomly from
    a set of corruptions which is determined by whether validation_corruptions is True or not. See function above too.
    :param batch_size: batch size of loader
    :param validation_corruptions: (bool) if True, it uses only the validation corruptions (the last 4 ones in dict d)
    :param severity: integer between 1 and 5 inclusive. 5 indicates high corruption
    :param train: train split or not (bool)
    :param device: device to load the data onto (maybe the device where the model is trained for speed?)
    :param sample_indices: range of indices of the datapoints to include (tuple of length 2). e.g. (20000, 30000) when
    Train = True, or (1000, 5000) is train = False.
    :return: data loader
    '''
    if not (0 <= severity <= 5):
        raise ValueError("Severity level must be in [1, 2, 3, 4, 5].")

    if sample_indices is not None:
        start_index = sample_indices[0]
        end_index = sample_indices[1]
    else:
        start_index = 0
        if train:
            end_index = 50000
        else:
            end_index = 10000

    corruption_list = corruption_choices(validation_corruptions)
    num_c = len(corruption_list)

    images_c = []
    for corruption in corruption_list:
        images_path = DATA_PATH + '/cifar10-C/' + d[
            corruption].__name__ + f'/severity_{severity}'
        labels_path = DATA_PATH + '/cifar10-C/' + d[
            corruption].__name__ + f'/severity_{severity}'

        images = np.load(
            images_path + f'/images_{"train" if train else "test"}.npy'
        )[start_index:end_index]
        images_c.append(images)

    images_c = np.stack(images_c)

    np.random.seed(severity) # Randomize a bit across severities!
    corruption_idxs = np.random.choice(range(num_c), end_index - start_index,
                                       replace=True)
    images = images_c[corruption_idxs, range(end_index - start_index)]
    labels = np.load(
        labels_path + f'/labels_{"train" if train else "test"}.npy'
    )[start_index:end_index]

    images = torch.stack([trn.ToTensor()(image) for image in images]).to(device)
    labels = torch.tensor(labels).to(device)

    # transforms are scale + center (with same mean, std as training data)
    trans = trn.Compose([trn.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = TensorDatasetWithTrans(tensors=(images, labels), transform=trans)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=False, pin_memory=False)

    return data_loader

