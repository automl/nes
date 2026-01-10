import torch
import torch.nn as nn
import numpy as np

from nes.nasbench201.bo.surrogate_models.modules import *


class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.device = torch.device('cpu')
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

    def fit(self, xtrain, ytrain,
            epochs=200,
            batch_size=32,
            lr=.01,
            **kwargs):

        #model = self.cuda()
        model = self.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.L1Loss().to(self.device)
        losses = []
        for _ in range(epochs):
            x, y = \
                torch.from_numpy(xtrain).float().to(self.device),\
                torch.from_numpy(ytrain).float().to(self.device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_error = losses[-1]
        return train_error

    def predict(self, xtest):
        x = torch.from_numpy(xtest).float().to(self.device)
        return self(x)


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))

    def fit(self, xtrain, ytrain,
            epochs=200,
            batch_size=32,
            lr=.01,
            **kwargs):

        model = self.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.L1Loss().cuda()
        losses = []
        for _ in range(epochs):
            x, y = \
                torch.from_numpy(xtrain).float().cuda(),\
                torch.from_numpy(ytrain).float().cuda()
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_error = losses[-1]
        return train_error

    def predict(self, xtest):
        x = torch.from_numpy(xtest).float().cuda()
        return self(x)

