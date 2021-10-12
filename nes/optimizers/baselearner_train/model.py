import os
import sys
import math
import json
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from nes.optimizers.baselearner_train.operations import *
from nes.optimizers.baselearner_train import genotypes


def get_init_std(p):
    if p.dim() == 1:
        return None
    fan = init._calculate_correct_fan(p, "fan_in")
    gain = init.calculate_gain("leaky_relu", math.sqrt(5))
    std = gain / math.sqrt(fan)
    return std


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction,
                 reduction_prev, drop_prob=0.3):
        super(Cell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)
        self.dropout = nn.Dropout2d(p=drop_prob)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):  # , drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training:  # and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    # h1 = drop_path(h1, drop_prob)
                    h1 = self.dropout(h1)
                if not isinstance(op2, Identity):
                    # h2 = drop_path(h2, drop_prob)
                    h2 = self.dropout(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype,
                 drop_prob=0.3):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                        reduction_prev, drop_prob)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)  # self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class DARTSByGenotype(nn.Module):
    def __init__(self, genotype, seed_init, dataset='fmnist', global_seed=1,
                 n_layers=8, init_channels=16, auxiliary_head=False,
                 drop_prob=0.3, **kwargs):
        super(DARTSByGenotype, self).__init__()

        assert dataset in ['cifar10', 'cifar100', 'fmnist', 'tiny']
        self.genotype = genotype
        layers = 8 if dataset in ['cifar10', 'cifar100', 'tiny'] else 5
        C = 16 if dataset in ['cifar10', 'cifar100', 'tiny'] else 8

        if dataset == 'tiny': num_classes = 200
        elif dataset == 'cifar100': num_classes = 100
        else: num_classes = 10

        self.num_classes = num_classes

        torch.manual_seed(seed_init + 400 * (global_seed - 1)) # TODO: global seed shouldn't be handled like this. what if we change max ensemble size (i.e. 30)?
        self.model = NetworkCIFAR(C=C, num_classes=num_classes, layers=layers,
                                  auxiliary=False, genotype=self.genotype,
                                  drop_prob=drop_prob)

    def forward(self, x):
        return self.model(x)[0]


    @classmethod
    def base_learner_train_save(cls, seed_init, arch_id, genotype, train_loader,
                                test_loader, num_epochs, save_path, device,
                                verbose=False, logger=None, dataset='fmnist',
                                debug=False, global_seed=1, lr=0.025,
                                wd=3e-4, n_layers=8, anchor=False, anch_coeff=1,
                                init_channels=16, drop_prob= 0.3, **kwargs):
        '''This function is the main training loop that trains and saves the
        model (at various checkpoints)'''

        if logger is None:
            raise ValueError("No logger provided.")

        learning_rate = lr
        print(lr, wd, anch_coeff)
        print('##################################')

        # Initialize architecture and weights using genotype and initialization seed.
        model = cls(genotype=genotype, seed_init=seed_init, dataset=dataset,
                    global_seed=global_seed, n_layers=n_layers,
                    init_channels=init_channels, drop_prob=drop_prob)
        model.to(device)

        # Loss and optimizer mostly using default settings from DARTS paper.
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=0.9, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               num_epochs)

        ###### ANCHORING SETUP
        if anchor:
            model_for_init = cls(genotype=genotype, seed_init=seed_init+13452,
                                 dataset=dataset, global_seed=global_seed+13452,
                                 n_layers=n_layers, init_channels=init_channels)
            model_for_init.to(device)

            anchor_params = [(p.data.clone(), get_init_std(p)) for p in
                             model_for_init.parameters()]
            assert all((not p.requires_grad) for p, _ in anchor_params)
        ###### END ANCHORING SETUP


        start_time = time.time()
        torch.manual_seed(0)
        total_step = len(train_loader)

        # Train the model
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):

                if images.device.type == "cpu" or labels.device.type == "cpu":
                    images = images.to(device)
                    labels = labels.to(device)

                outputs = model(images)
                outputs = outputs.log_softmax(1)
                loss_nll = criterion(outputs, labels)

                ### START ANCHORED ###
                if anchor:
                    anch_reg = 0
                    # leave out the weight and bias of the final fc layer
                    for p, (p_anch, init_std) in zip(list(model.parameters())[:-2], anchor_params[:-2]):
                        assert p.dim() in [1, 4]
                        if p.dim() == 1: # skip batch norm params
                            continue

                        anch_reg += (1 / (2 * init_std**2)) * (p - p_anch).pow(2).sum()

                    # final layer weight
                    for p, (p_anch, init_std) in zip(list(model.parameters())[-2:-1], anchor_params[-2:-1]):
                        assert p.dim() in [2]
                        anch_reg += (1 / (2 * init_std**2)) * (p - p_anch).pow(2).sum()

                    # p is the weight matrix of last layer
                    fan_in, _ = init._calculate_fan_in_and_fan_out(p)
                    # std of init of bias of last layer
                    init_std = 1 / math.sqrt(3 * fan_in)

                    # final layer bias
                    for p, (p_anch, _) in zip(list(model.parameters())[-1:], anchor_params[-1:]):
                        assert p.dim() in [1]
                        assert p.shape[0] == model.num_classes
                        anch_reg += (1 / (2 * init_std**2)) * (p - p_anch).pow(2).sum()

                    anch_reg = anch_reg / len(train_loader.dataset) # num_datapoints
                    loss = loss_nll + anch_coeff * anch_reg
                    anch_reg_to_print = anch_reg.item()
                else:
                    loss = loss_nll
                    anch_reg_to_print = 0
                ### END ANCHORED ###

                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()

                if torch.isnan(loss):
                    raise ValueError("Training failed. Loss is NaN.")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if debug:
                    break

                if verbose:
                    if (i + 1) % 100 == 0:
                        logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Anch_loss: {:.4f}, Accuracy: {:.4f}. Model_ID: (arch {}, init {})'.format(epoch + 1, num_epochs, i + 1, total_step, loss_nll.item(), anch_reg_to_print, correct / total, arch_id, seed_init))

            scheduler.step()
            if debug:
                break

        logger.info('Training completed for model (arch {}, init {}) in {} '
                    'secs.'.format(arch_id, seed_init, round(time.time() -
                    start_time, 2)))

        # save model checkpoint
        model_save_path = os.path.join(
            save_path, f"arch_{arch_id}_init_{seed_init}_epoch_{num_epochs}.pt"
        )
        torch.save(model.state_dict(), model_save_path)
        logger.info(
            'Saved model (arch {}, init {}) after epoch {} in {} '
            'secs.'.format(arch_id, seed_init, num_epochs, round(time.time()
            - start_time, 2)))

        return model
