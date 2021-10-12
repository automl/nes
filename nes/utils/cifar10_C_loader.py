import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from PIL import Image

from data.corruptions import *
from nes.utils.nb201.DownsampledImageNet import ImageNet16


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


# =========================================== #
# Data loaders for CIFAR-10-C and CIFAR-100-C #
# =========================================== #
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


def build_dataloader_cifar_c(batch_size, corruption, severity, train, device,
                             sample_indices=None, dataset='cifar10',
                             nb201=False):
    '''
    Builds a CIFAR-10 or CIFAR-100 dataloader of corrupted images. The
        corruption type and severity are inputs.
    :param batch_size: batch size of loader
    :param corruption: corruption as string. has to be one of the strings in
        the dictionary d above.
    :param severity: integer between 1 and 5 inclusive. 5 indicates high
        corruptiom
    :param train: train split or not (bool)
    :param device: device to load the data onto (maybe the device where the
        model is trained for speed?)
    :param sample_indices: range of indices of the datapoints to include (tuple
        of length 2). e.g. (20000, 30000) when Train = True, or (1000, 5000) is
        train = False.
    :return: data loader
    '''
    if not (0 <= severity <= 5):
        raise ValueError("Severity level must be in [1, 2, 3, 4, 5].")

    assert dataset in ['cifar10', 'cifar100']

    corruption_choices = ['Gaussian Noise', 'Shot Noise', 'Impulse Noise',
                          'Defocus Blur', 'Glass Blur', 'Motion Blur',
                          'Zoom Blur', 'Snow', 'Frost', 'Fog', 'Brightness',
                          'Contrast', 'Elastic', 'Pixelate', 'JPEG',
                          'Speckle Noise', 'Gaussian Blur', 'Spatter',
                          'Saturate']

    if not corruption in (corruption_choices + ["clean"]):
        raise ValueError(f"No corruption of type {corruption}.")

    if sample_indices is None:
        start_index = 0
        if train:
            end_index = 50000
        else:
            end_index = 10000
        sample_indices = list(range(start_index, end_index))

    if corruption != "clean":
        images_path = DATA_PATH + '/' + dataset + '-C/' + d[
            corruption].__name__ + f'/severity_{severity}'
        labels_path = DATA_PATH + '/' + dataset + '-C/' + d[
            corruption].__name__ + f'/severity_{severity}'

        images = np.load(
            images_path + f'/images_{"train" if train else "test"}.npy'
        )[sample_indices, ...]

        labels = np.load(
            labels_path + f'/labels_{"train" if train else "test"}.npy'
        )[sample_indices, ...]
    else:
        dataset_cls = dset.CIFAR10 if dataset == 'cifar10' else dset.CIFAR100
        _dataset = dataset_cls(DATA_PATH, train=train, download=True)
        images = _dataset.data
        labels = np.array(_dataset.targets)

        images = images[sample_indices, ...]
        labels = labels[sample_indices, ...]

    images = torch.stack([trn.ToTensor()(image) for image in images]).to(device)
    labels = torch.tensor(labels).to(device)

    # transforms are scale + center (with same mean, std as training data)
    if not nb201:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        if dataset == 'cifar10':
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std  = [x / 255 for x in [63.0, 62.1, 66.7]]
        elif dataset == 'cifar100':
            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]

    trans = trn.Compose([trn.Normalize(mean, std)])
    dataset = TensorDatasetWithTrans(tensors=(images, labels), transform=trans)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=False, pin_memory=False)
    return data_loader


def build_dataloader_cifar_c_mixed(batch_size, validation_corruptions,
                                   severity, train, device,
                                   sample_indices=None, dataset='cifar10',
                                   nb201=False):
    '''
    Builds a CIFAR-10 or CIFAR-100 dataloader of corrupted images. The
        corruption for each datapoint is picked randomly from a set of corruptions
        which is determined by whether validation_corruptions is True or not. See
        function above too.
    :param batch_size: batch size of loader
    :param validation_corruptions: (bool) if True, it uses only the validation
        corruptions (the last 4 ones in dict d)
    :param severity: integer between 1 and 5 inclusive. 5 indicates high
        corruption
    :param train: train split or not (bool)
    :param device: device to load the data onto (maybe the device where the
        model is trained for speed?)
    :param sample_indices: range of indices of the datapoints to include (tuple
        of length 2). e.g. (20000, 30000) when Train = True, or (1000, 5000) is
        train = False.
    :return: data loader
    '''
    if not (0 <= severity <= 5):
        raise ValueError("Severity level must be in [1, 2, 3, 4, 5].")

    assert dataset in ['cifar10', 'cifar100']

    if sample_indices is None:
        start_index = 0
        if train:
            end_index = 50000
        else:
            end_index = 10000
        sample_indices = list(range(start_index, end_index))

    corruption_list = corruption_choices(validation_corruptions)
    num_c = len(corruption_list)

    images_c = []
    for corruption in corruption_list:
        images_path = DATA_PATH + '/' + dataset + '-C/' + d[
            corruption].__name__ + f'/severity_{severity}'
        labels_path = DATA_PATH + '/' + dataset + '-C/' + d[
            corruption].__name__ + f'/severity_{severity}'

        images = np.load(
            images_path + f'/images_{"train" if train else "test"}.npy'
        )[sample_indices, ...]
        images_c.append(images)

    images_c = np.stack(images_c)

    np.random.seed(severity) # Randomize a bit across severities!
    corruption_idxs = np.random.choice(range(num_c), len(sample_indices),
                                       replace=True)
    images = images_c[corruption_idxs, range(len(sample_indices))]
    labels = np.load(
        labels_path + f'/labels_{"train" if train else "test"}.npy'
    )[sample_indices, ...]

    images = torch.stack([trn.ToTensor()(image) for image in images]).to(device)
    labels = torch.tensor(labels).to(device)

    # transforms are scale + center (with same mean, std as training data)
    if not nb201:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        if dataset == 'cifar10':
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std  = [x / 255 for x in [63.0, 62.1, 66.7]]
        elif dataset == 'cifar100':
            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]

    trans = trn.Compose([trn.Normalize(mean, std)])
    dataset = TensorDatasetWithTrans(tensors=(images, labels), transform=trans)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=False, pin_memory=False)
    return data_loader


def build_dataloader_tiny(batch_size, severity, mode, n_workers=4):
    '''
    Builds a Tiny-ImageNet dataloader of corrupted images. The corruption type
        and severity are inputs.
    :param batch_size: batch size of loader
    :param severity: integer between 1 and 5 inclusive. 5 indicates high
        corruption
    :mode: train, val or test
    :return: data loader
    '''
    if not (0 <= severity <= 5):
        raise ValueError("Severity level must be in [1, 2, 3, 4, 5].")

    data_path = os.path.join(DATA_PATH, 'tiny-imagenet-200')
    if severity > 0: data_path += '-c-out'
    data_path = os.path.join(data_path, mode)
    if severity > 0: data_path = os.path.join(data_path,
                                              f'severity_{severity}')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == 'train':
        transforms = trn.Compose([trn.RandomResizedCrop(64),
                                  trn.RandomHorizontalFlip(),
                                  trn.ColorJutter(
                                      brightness=0.4,
                                      contrast=0.4,
                                      saturation=0.4,
                                      hue=0.2),
                                  trn.ToTensor(),
                                  trn.Normalize(mean, std)])
        shuffle = True
    else:
        transforms = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        shuffle = False

    dataset = dset.ImageFolder(root=data_path, transform=transforms)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=n_workers,
                             pin_memory=True)
    return data_loader


def build_dataloader_imagenet(batch_size, corruption, severity, train, device,
                              sample_indices=None, dataset='imagenet',
                              nb201=False):
    '''
    Builds the dataloader for the downsampled ImageNet-16-120 from the
        NAS-Bench-201 paper.
    :param batch_size: batch size of loader
    :param corruption: corruption as string. has to be one of the strings in the dictionary d above.
    :param severity: integer between 1 and 5 inclusive. 5 indicates high corruption
    :param train: train split or not (bool)
    :param device: device to load the data onto (maybe the device where the model is trained for speed?)
    :param sample_indices: range of indices of the datapoints to include (tuple of length 2). e.g. (20000, 30000) when
    Train = True, or (1000, 5000) is train = False.
    :return: data loader
    '''
    _dataset = ImageNet16(os.path.join(DATA_PATH, "imagenet_resized_16"),
                          False, None, 120)
    assert len(_dataset) == 6000
    images = np.array(_dataset.data)
    labels = np.array(_dataset.targets)

    images = images[sample_indices, ...]
    labels = labels[sample_indices, ...]

    images = torch.stack([trn.ToTensor()(image) for image in images]).to(device)
    labels = torch.tensor(labels).to(device)
    labels -= 1 # from 1...120 to 0...119

    # transforms are scale + center (with same mean, std as training data)
    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std = [x / 255 for x in [63.22,  61.26 , 65.09]]

    trans = trn.Compose([trn.Normalize(mean, std)])
    dataset = TensorDatasetWithTrans(tensors=(images, labels), transform=trans)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=False, pin_memory=False)
    return data_loader



