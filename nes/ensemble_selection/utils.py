import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import TensorDataset
from collections import defaultdict

from nes.utils.data_loaders import (
    build_dataloader_cifar_c_mixed,
    build_dataloader_cifar_c,
    build_dataloader_imagenet,
    build_dataloader_tiny,
    build_dataloader_fmnist
)


def make_predictions(models, data_loader, device, num_classes):
    if not (isinstance(models, dict) or isinstance(models, nn.Module)):
        raise TypeError("models must be a dictionary or nn.Module.")

    got_nn_Module = False
    if isinstance(models, nn.Module):
        models = {"0": models}
        got_nn_Module = True

    N = len(models)

    batch_size = data_loader.batch_size
    num_points = len(data_loader.dataset)

    models_predictions = torch.empty(num_points, N, num_classes, device=device)
    all_labels = torch.empty(num_points, dtype=torch.int64, device=device)

    for n, net in enumerate(models.keys()):
        model = models[net]
        was_training = model.training
        model.to(device)
        model.eval()
        with torch.no_grad():
            for index, (images, labels) in enumerate(data_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if len(outputs) == 2:
                    # nb201 case
                    outputs = outputs[1]

                batch_start_index = index * batch_size
                batch_end_index = min((index + 1) * batch_size, num_points)

                all_labels[batch_start_index:batch_end_index] = labels
                models_predictions[batch_start_index:batch_end_index, n, :] = outputs

        if was_training:
            model.train()
        if N > 1:
            print("Computed model " + str(net) + " predictions on dataset.")

    if got_nn_Module:
        models_predictions = torch.squeeze(models_predictions)

    pred_dataset = TensorDataset(models_predictions, all_labels)
    return pred_dataset


def compute_ece(preds, labels):
    resolution = 11 # number of bins 
    total_in_bin = [0 for _ in range(resolution - 1)]
    correct_in_bin = [0 for _ in range(resolution - 1)]
    confidences_in_bin = [[] for _ in range(resolution - 1)]

    bins = list(np.linspace(0, 1, resolution))
    up_list = bins[1:]
    low_list = bins[:-1]

    confidences, predicted = torch.max(preds.exp().data, 1)
    accuracies = predicted.eq(labels)

    for j, (bin_lower, bin_upper) in enumerate(zip(low_list, up_list)):
        in_bin = confidences.gt(bin_lower) & confidences.le(bin_upper)
        num_in_bin = in_bin.float().sum()
        if num_in_bin > 0:
            total_in_bin[j] += num_in_bin.item()
            correct_in_bin[j] += accuracies[in_bin].float().sum().item()
            confidences_in_bin[j].extend(confidences[in_bin].tolist())

    correct_in_bin = [
        elem for index, elem in enumerate(correct_in_bin) if total_in_bin[index] > 0
    ]
    confidences_in_bin = [
        elem for index, elem in enumerate(confidences_in_bin) if total_in_bin[index] > 0
    ]
    total_in_bin = [elem for elem in total_in_bin if elem > 0]

    assert len(correct_in_bin) == len(confidences_in_bin) == len(total_in_bin)

    prop_in_bin = [a / sum(total_in_bin) for a in total_in_bin]
    acc_in_bin = [a / b for a, b in zip(correct_in_bin, total_in_bin)]
    avg_confidence_in_bin = [np.mean(elem) for elem in confidences_in_bin]

    ece = sum(
        [
            abs(a - b) * c
            for a, b, c in zip(avg_confidence_in_bin, acc_in_bin, prop_in_bin)
        ]
    )

    return ece


def classif_accuracy(outputs, labels):
    total = labels.size(0)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return correct / total


def evaluate_predictions(preds_dataset, lsm_applied=False):
    nll = nn.NLLLoss()

    preds = preds_dataset.tensors[0]
    labels = preds_dataset.tensors[1]

    if not lsm_applied:
        preds = preds.log_softmax(1)

    loss = nll(preds, labels)
    acc = classif_accuracy(preds, labels)
    ece = compute_ece(preds, labels)

    return {"loss": loss.item(), "acc": acc, "ece": ece}


def form_ensemble_pred(
    models_preds_tensors, lsm_applied=False, combine_post_softmax=True, bsl_weights=None,
):
    assert isinstance(models_preds_tensors, dict)

    if combine_post_softmax:
        if not lsm_applied:
            all_outputs = torch.stack(
                [models_preds_tensors[net].softmax(1) for net in models_preds_tensors],
                dim=1,
            )
        else:
            all_outputs = torch.stack(
                [models_preds_tensors[net].exp() for net in models_preds_tensors], dim=1
            )
        # import ipdb; ipdb.set_trace()
        if bsl_weights is None:
            avg_output = torch.mean(all_outputs, dim=1) # simple unweighted average
        else:
            #print(bsl_weights.sum())
            assert bsl_weights.shape == (len(models_preds_tensors),)
            #assert bsl_weights.sum() == 1, "Weights don't sum to one."
            avg_output = torch.matmul(bsl_weights.unsqueeze(0), all_outputs).squeeze(1)

        return avg_output.log()

    else:
        assert bsl_weights is None, "Not implemented."
        if lsm_applied:
            raise ValueError(
                "Cannot combine baselearner predictions pre-softmax when softmax is already applied."
            )

        all_outputs = torch.stack(
            [models_preds_tensors[net] for net in models_preds_tensors], dim=1
        )
        avg_output = torch.mean(all_outputs, dim=1)

        return avg_output.log_softmax(1)


def form_ensemble_pred_v_2(models_preds_tensors, lsm_applied=False, combine_post_softmax=True):
    "the above version doesn't work when you have repeated base learners..."

    assert isinstance(models_preds_tensors, list)

    if combine_post_softmax:
        if not lsm_applied:
            all_outputs = torch.stack([preds.softmax(1) for preds in models_preds_tensors], dim=1)
        else:
            all_outputs = torch.stack([preds.exp() for preds in models_preds_tensors], dim=1)
        avg_output = torch.mean(all_outputs, dim=1)

        return avg_output.log()

    else:
        if lsm_applied:
            raise ValueError('Cannot combine baselearner predictions pre-softmax when softmax is already applied.')

        all_outputs = torch.stack(models_preds_tensors, dim=1)
        avg_output = torch.mean(all_outputs, dim=1)

        return avg_output.log_softmax(1)


# thanks to Bryn Elesedy
class Registry:
    def __init__(self):
        self._register = dict()

    def __getitem__(self, key):
        return self._register[key]

    def __call__(self, name):
        def add(thing):
            self._register[name] = thing
            return thing

        return add


def args_to_device(arg_device):
    return (
        torch.device(f"cuda:{arg_device}") if arg_device != -1 else torch.device("cpu")
    )


def get_indices(path='nes/utils/nb201/configs', dataset='cifar10'):
    if dataset == 'cifar10':
        filename = 'cifar-split.txt'
        split_1 = 'train'
        split_2 = 'valid'
    elif dataset == 'cifar100':
        filename = 'cifar100-test-split.txt'
        split_1 = 'xtest'
        split_2 = 'xvalid'
    elif dataset == 'imagenet':
        filename = 'imagenet-16-120-test-split.txt'
        split_1 = 'xtest'
        split_2 = 'xvalid'

    with open(os.path.join(path, filename)) as f:
        split = eval(f.read())

    # for C100 and imagenet is indices_test actually
    indices_train = list(map(int, split[split_1][1]))
    indices_valid = list(map(int, split[split_2][1]))
    return indices_train, indices_valid


def create_dataloader_dict_imagenet(device):
    dataloaders = defaultdict(dict)
    samples_test, samples_valid = get_indices(dataset='imagenet')

    dataloaders["val"]['0'] = build_dataloader_imagenet(
        batch_size=100,
        corruption="clean",
        severity=0,
        train=False,
        device=device,
        sample_indices=samples_valid,
        dataset='imagenet',
    )
    dataloaders["test"]['0'] = build_dataloader_imagenet(
        batch_size=100,
        corruption="clean",
        severity=0,
        train=False,
        device=device,
        sample_indices=samples_test,
        dataset='imagenet',
    )
    dataloaders["metadata"] = {"device": device}
    return dataloaders


def create_dataloader_dict_cifar(device, dataset='cifar10', nb201=False,
                                 n_datapoints=None):
    dataloaders = defaultdict(dict)

    if nb201:
        samples_train, samples_valid = get_indices(dataset=dataset)
        if dataset == 'cifar10':
            train = True
            samples_test = list(range(10000))
        elif dataset == 'cifar100':
            train = False
            samples_test = samples_train
    else:
        train = True
        samples_test = list(range(10000))
        if n_datapoints is None:
            samples_valid = list(range(40000, 50000))
        else:
            samples_valid = list(range(n_datapoints, 50000))

    for severity in range(0, 6):
        if severity == 0:
            dataloaders["val"][str(severity)] = build_dataloader_cifar_c(
                batch_size=100,
                corruption="clean",
                severity=0,
                train=train,
                device=device,
                sample_indices=samples_valid,
                dataset=dataset,
                nb201=nb201,
            )
            dataloaders["test"][str(severity)] = build_dataloader_cifar_c(
                batch_size=100,
                corruption="clean",
                severity=0,
                train=False,
                device=device,
                sample_indices=samples_test,
                dataset=dataset,
                nb201=nb201,
            )
        else:
            dataloaders["val"][str(severity)] = build_dataloader_cifar_c_mixed(
                100, True, severity, train, device, samples_valid, dataset,
                nb201
            )
            dataloaders["test"][str(severity)] = build_dataloader_cifar_c_mixed(
                100, False, severity, False, device, samples_test, dataset,
                nb201
            )
    dataloaders["metadata"] = {"device": device}
    return dataloaders


def create_dataloader_dict_tiny(device):
    dataloaders = defaultdict(dict)

    for severity in range(0, 6):
        for mode in ["val", "test"]:
            dataloaders[mode][str(severity)] = build_dataloader_tiny(100,
                                                                     severity,
                                                                     mode)
    dataloaders["metadata"] = {"device": device}
    return dataloaders


def create_dataloader_dict_fmnist(device):
    dataloaders = defaultdict(dict)
    dataloaders["val"][str(0)] = build_dataloader_fmnist(
        batch_size=100,
        train=True,
        device=device,
        sample_indices=list(range(50000, 60000))
    )
    dataloaders["test"][str(0)] = build_dataloader_fmnist(
        batch_size=100,
        train=False,
        device=device,
        sample_indices=list(range(0, 10000))
    )
    dataloaders["metadata"] = {"device": device}
    return dataloaders


def get_dataloaders_and_classes(dataset, device, nb201, n_datapoints):
    if dataset == 'fmnist':
        dataloaders = create_dataloader_dict_fmnist(device)
    elif dataset in ['cifar10', 'cifar100']:
        dataloaders = create_dataloader_dict_cifar(device, dataset, nb201,
                                                   n_datapoints)
    elif dataset == 'imagenet':
        dataloaders = create_dataloader_dict_imagenet(device)
    elif dataset == 'tiny':
        dataloaders = create_dataloader_dict_tiny(device)

    if dataset == 'imagenet':
        num_classes = 120
    elif dataset == 'tiny':
        num_classes = 200
    else:
        num_classes = 100 if dataset == 'cifar100' else 10

    return dataloaders, num_classes


