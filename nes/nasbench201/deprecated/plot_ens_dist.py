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

#plt.scatter(test_preds, ytest)
#plt.xlim([0.8, 1.6])
#plt.ylim([0.8, 1.6])
#plt.xlabel('Predited Ensemble NLL')
#plt.ylabel('True Ensemble NLL')
#plt.grid()
#plt.title('Kendall Tau: %.4f'%tau)
#plt.savefig('plots/scatter_%d'%xsize)
#plt.close()
