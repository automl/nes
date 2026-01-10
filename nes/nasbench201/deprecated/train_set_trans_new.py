import os
import random
import warnings
import numpy as np
import pickle
import torch
warnings.filterwarnings('ignore')

from nes.nasbench201.bo.surrogate_models.set_transformer import DeepSet, SetTransformer
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, norm


def plot_predictions(mu_train, mu_test, var_train, var_test, train_y, test_y,
                     tau_train, tau_test, save_dir,
                     name='random forest'):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    f, ax = plt.subplots(1, 2, figsize=(15, 6))

    if var_train is not None:
        ll = norm.logpdf(np.squeeze(train_y), loc=mu_train, scale=np.sqrt(var_train))
        c_map = 'viridis'
    else:
        ll = 'b'
        c_map = None

    im1 = ax[0].scatter(mu_train, train_y, c=ll, cmap=c_map)
    ax[0].set_xlabel('Predicted Ensemble NLL', fontsize=15)
    ax[0].set_ylabel('True Ensemble NLL', fontsize=15)
    ax[0].set_title('{} (train), Kendall Tau: {:.2f}'.format(name, tau_train), fontsize=11)
    ax[0].plot([0.75, 1.6], [0.75, 1.6], 'k--')
    ax[0].grid()
    if var_train is not None:
        f.colorbar(im1, ax=ax[0])

    if var_test is not None:
        ll = norm.logpdf(np.squeeze(test_y), loc=mu_test, scale=np.sqrt(var_test))
        c_map = 'viridis'
    else:
        ll = 'b'
        c_map = None

    im1 = ax[1].scatter(mu_test, test_y, c=ll, cmap=c_map)
    ax[1].set_xlabel('Predicted Ensemble NLL', fontsize=15)
    ax[1].set_ylabel('True Ensemble NLL', fontsize=15)
    ax[1].set_title('{} (test), Kendall Tau: {:.2f}'.format(name, tau_test), fontsize=11)
    ax[1].plot([0.75, 1.6], [0.75, 1.6], 'k--')
    ax[1].grid()
    if var_test is not None:
        f.colorbar(im1, ax=ax[1])
    plt.tight_layout()
    plt.savefig(save_dir + '/' + '_'.join(name.split()) + '.pdf')
    plt.close()


def main(args):
    save_dir = args.save_dir #'plots3'
    model_name = args.model_name #'DeepSet'
    num_hidden = args.num_hidden #512
    epochs = args.epochs

    with open('surr_data.pkl', 'rb') as f:
        query_data = pickle.load(f)

    dx = np.array(query_data['x'])
    dy = np.array(query_data['y']).reshape(-1, 1)

    #xtest = dx[45000:,...]
    #ytest = dy[45000:,...]

    #xtest = np.array([np.array_split(i, 5) for i in xtest])
    dx = np.array([np.array_split(i, 5) for i in dx])
    excluded_arch = dx[0][0].tolist()
    print('Excluded architecture: {}'.format(excluded_arch))
    occurrences = np.array([excluded_arch in ens for ens in dx.tolist()])
    xtest = dx[occurrences]
    ytest = dy[occurrences]
    print('Test set size: {}'.format(len(xtest)))
    _xtrain = dx[~occurrences]
    _ytrain = dy[~occurrences]

    for xsize in [100, 500, 1000, 5000, 10000, 20000, 30000]:
        print('Size of training data: %d'%xsize)
        xtrain = _xtrain[:xsize,...]
        ytrain = np.expand_dims(_ytrain[:xsize,...], axis=1)

        train_candidate_preds = []
        test_candidate_preds = []
        train_error = 0
        test_error = 0
        for _ in range(3):
            surr = eval(model_name)(dim_input=6, num_outputs=1, dim_output=1,
                                    dim_hidden=num_hidden)
            train_error += surr.fit(xtrain, ytrain, epochs=epochs)

            train_pred = np.squeeze(surr.predict(xtrain).detach().cpu().numpy())
            train_candidate_preds.append(train_pred)
            test_pred = np.squeeze(surr.predict(xtest).detach().cpu().numpy())
            test_candidate_preds.append(test_pred)

            test_error += np.mean(abs(test_pred-ytest))

        train_error /= 3
        print('Surrogate train error: {}'.format(train_error))

        test_error /= 3
        print('Surrogate test error: {}'.format(test_error))

        train_mu = np.mean(train_candidate_preds, axis=0)
        train_var = np.var(train_candidate_preds, axis=0)
        test_mu = np.mean(test_candidate_preds, axis=0)
        test_var = np.var(test_candidate_preds, axis=0)

        tau_train, rho = kendalltau(train_mu, ytrain)
        print(tau_train, rho)
        tau_test, rho = kendalltau(test_mu, ytest)
        print(tau_test, rho)

        plot_predictions(train_mu, test_mu, train_var, test_var, ytrain, ytest,
                         tau_train, tau_test, save_dir,
                         name="{} (dim_hidden={},epochs={},"
                         "train_size={})".format(model_name, num_hidden, epochs, xsize))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DeepSet',
                        help='Surrogate model name')
    parser.add_argument('--num_hidden', type=int,
                        help='hidden layer size.',
                        default=512)
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train surrogate')
    parser.add_argument('--save_dir', type=str, default='plots3',
                        help='save directory')
    args = parser.parse_args()

    main(args)

