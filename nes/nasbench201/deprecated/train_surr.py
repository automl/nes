import os
import random
import warnings
import numpy as np
import pickle
warnings.filterwarnings('ignore')

from nes.nasbench201.bo.surrogate_models.bananas_mlp import MetaNeuralnet
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

if not os.path.exists('plots'):
    os.makedirs('plots')

with open('surr_data.pkl', 'rb') as f:
    query_data = pickle.load(f)


dx = np.array(query_data['x'])
dy = np.array(query_data['y']).reshape(-1, 1)

xtest = dx[45000:,...]
ytest = dy[45000:,...]

for xsize in [100, 500, 1000, 5000, 10000, 20000, 30000]:
    print('Size of training data: %d'%xsize)
    xtrain = dx[:xsize,...]
    ytrain = dy[:xsize,...]
    candidate_preds = []
    train_error = 0
    test_error = 0
    for _ in range(3):
        surr = MetaNeuralnet()
        train_error += surr.fit(xtrain, ytrain)

        test_pred = np.squeeze(surr.predict(xtest))
        candidate_preds.append(test_pred)
        test_error += np.mean(abs(test_pred-ytest))
        print(test_error)

        # clear the tf graph. Used for the bananas surrogate
        tf.reset_default_graph()

    tf.keras.backend.clear_session()

    train_error /= 3
    print('Meta neural net train error: {}'.format(train_error))

    test_error /= 3
    print('Meta neural net test error: {}'.format(test_error))

    test_preds = np.mean(candidate_preds, axis=0)

    tau, rho = kendalltau(test_preds, ytest)
    print(tau, rho)

    plt.scatter(test_preds, ytest)
    plt.xlim([0.8, 1.6])
    plt.ylim([0.8, 1.6])
    plt.xlabel('Predited Ensemble NLL')
    plt.ylabel('True Ensemble NLL')
    plt.grid()
    plt.title('Kendall Tau: %.4f'%tau)
    plt.savefig('plots/scatter_%d'%xsize)
    plt.close()
