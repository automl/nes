import torch
import os
import pickle
from tqdm import tqdm
import sys
import multiprocessing
import numpy as np

PATH = '/data/aad/image_datasets/nb201_new/NAS-BENCH-102-4-v1.0-archive/'

sorted_by_idx = sorted(os.listdir(PATH))

def load_pkl(pkl_name):
    with open(pkl_name+'.pkl', 'rb') as f:
        a = pickle.load(f)
    return a

def save_pkl(list_of_ids, pkl_name):
    with open(pkl_name + '.pkl', 'ab') as f:
        pickle.dump(list_of_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

# load arch indices with 3 seeds
c10 = load_pkl('cifar10')
c100 = load_pkl('cifar100')
imagenet = load_pkl('imagenet')

d_dict = {
    'cifar10-valid': c10,
    'cifar100': c100,
    'ImageNet16-120': imagenet,
}

opt_dict = {
    'cifar10-valid': float('inf'),
    'cifar100': float('inf'),
    'ImageNet16-120': float('inf'),
}


opt_dict_id = {
    'cifar10-valid': None,
    'cifar100': None,
    'ImageNet16-120': None,
}

def multiprocessing_func(start, end):
    for m in tqdm(sorted_by_idx[start: end]):
        xdata = torch.load(PATH + m)
        info = xdata['full']['all_results']
        for d in d_dict.keys():
            if xdata['full']['arch_index'] in d_dict[d]:
                mean_loss = np.mean([
                    list(info[(d, i)]['eval_losses'].values())[-1] for i in [777,888,999]
                ])
                if mean_loss < opt_dict[d]:
                    opt_dict[d] = mean_loss
                    opt_dict_id[d] = xdata['full']['arch_index']

if __name__ == '__main__':
    multiprocessing_func(0, len(sorted_by_idx))

    #for d, l in d_dict.items():
    #    save_pkl(l, d)
    with open('optimal.txt', 'w') as f:
        f.write(str(opt_dict_id))
