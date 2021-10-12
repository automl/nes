import torch
import os
import pickle
from tqdm import tqdm
import sys
import multiprocessing

PATH = '/data/aad/image_datasets/nb201_new/NAS-BENCH-102-4-v1.0-archive/'

sorted_by_idx = sorted(os.listdir(PATH))


def save_pkl(list_of_ids, pkl_name):
    with open(pkl_name + '.pkl', 'ab') as f:
        pickle.dump(list_of_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

d_dict = dict()


def multiprocessing_func(start, end):
    for m in tqdm(sorted_by_idx[start: end]):
        xdata = torch.load(PATH + m)
        key = xdata['full']['arch_str']
        d_dict[key] = (xdata['full']['arch_index'], m)


if __name__ == '__main__':
    #processes = []
    #for i in [(0, 2500), (2500, 5000), (5000, 7500), (7500, 10000), (10000,
    #                                                                 12500),
    #          (12500, len(sorted_by_idx))]:
    #    p = multiprocessing.Process(target=multiprocessing_func, args=(i[0],
    #                                                                   i[1]))
    #    processes.append(p)
    #    p.start()

    #for process in processes:
    #    process.join()
    multiprocessing_func(0, len(sorted_by_idx))
    save_pkl(d_dict, 'arch_to_id')
