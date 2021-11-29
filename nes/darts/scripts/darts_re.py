import os
import argparse
import logging
import pickle
import time
import numpy as np
import torch

import hpbandster.core.result as hputil
from hpbandster.core.nameserver import NameServer, nic_name_to_host
from hpbandster.utils import *
from ConfigSpace.read_and_write import json as config_space_json_r_w

from nes.darts.cluster_worker import DARTSWorker as Worker
from nes.darts.re import RegularizedEvolution


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

parser = argparse.ArgumentParser(description='The NES-RE algorithm in parallel.')
parser.add_argument('--num_iterations', type=int, help='number of function evaluations performed.', default=400)
parser.add_argument('--run_id', type=int, default=0)
parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.', default='eth0')
parser.add_argument('--working_directory', type=str, help='directory where to store the live rundata', default=None)
parser.add_argument('--array_id', type=int, default=1)
parser.add_argument('--total_num_workers', type=int, default=20)
parser.add_argument('--global_seed', type=int, default=1, help='Seed')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to use.')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--dataset', type=str, default='fmnist', help='dataset')
parser.add_argument('--warmstart_dir', type=str, default=None, help='warmstart directory')
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode')
parser.add_argument('--population_size', type=int, default=50, help='The population size to consider.')
parser.add_argument('--sample_size', type=int, default=10, help='The ensemble size to consider.')
parser.add_argument('--scheme', type=str, default='nes_re', help='scheme name')
parser.add_argument('--severity_list', type=str, default='0 5',
                    help='Severity levels to sample from during evolution')
parser.add_argument('--esa', type=str, default='beam_search',
                    help='Ensemble selection algorithm')
parser.add_argument('--lr', type=float, default=0.025,
                    help='leaning rate to train the baselearner')
parser.add_argument('--wd', type=float, default=3e-4,
                    help='weight decay to train the baselearner')
parser.add_argument('--n_layers', type=int, default=8,
                    help='Mini-batch size to train the baselearner')
parser.add_argument('--init_channels', type=int, default=16,
                    help='Mini-batch size to train the baselearner')
parser.add_argument('--scheduler', type=str, default='cosine',
                    help='scheme name, i.e. nes or deepens variants')
parser.add_argument('--n_workers', type=int, default=4,
                    help='Number of CPU workers')
parser.add_argument('--grad_clip', action='store_true', default=False,
                    help='debug mode: run for a single mini-batch')

args = parser.parse_args()

assert args.global_seed > 0, "global seed should be greater than 0"
np.random.seed(args.global_seed)
torch.manual_seed(args.global_seed)

args.working_directory = os.path.join(args.working_directory,
                                      'run_%d'%args.global_seed)

host = nic_name_to_host(args.nic_name)


if args.array_id == 1:
    os.makedirs(args.working_directory, exist_ok=True)
    with open(os.path.join(args.working_directory, 'settings.txt'), 'w') as f:
        f.write(str(args))

    NS = NameServer(run_id=args.run_id, host=host,
                    working_directory=args.working_directory)
    ns_host, ns_port = NS.start()

    # Regularized Evolution is usually so cheap, that we can afford to run a
    # worker on the master node as a background process
    worker = Worker(nameserver=ns_host,
                    nameserver_port=ns_port,
                    host=host,
                    run_id=args.run_id,
                    working_directory=args.working_directory,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    wd=args.wd,
                    scheme=args.scheme,
                    warmstart_dir=args.warmstart_dir,
                    dataset=args.dataset,
                    n_workers=args.n_workers,
                    nb201=False,
                    debug=args.debug)
    worker.run(background=True)

    # instantiate Regularized Evolution and run it
    result_logger = hputil.json_result_logger(directory=args.working_directory,
                                              overwrite=True)

    # fmnist does not have corruptions
    if args.dataset == 'fmnist':
        args.severity_list = '0'

    # instantiate the master
    algo = RegularizedEvolution(configspace=worker.get_configspace(),
                                host=host,
                                run_id=args.run_id,
                                nameserver=ns_host,
                                nameserver_port=ns_port,
                                min_budget=args.num_epochs,
                                max_budget=args.num_epochs,
                                population_size=args.population_size,
                                pop_sample_size=args.sample_size,
                                scheme=args.scheme,
                                esa=args.esa,
                                warmstart_dir=args.warmstart_dir,
                                severity_list=[int(x) for x in
                                               args.severity_list.split()],
                                working_directory=args.working_directory,
                                ping_interval=3600,
                                previous_result=None,
                                result_logger=result_logger)

    # hpbandster can wait until a minimum number of workers is online before starting
    res = algo.run(n_iterations=args.num_iterations,
                   min_n_workers=args.total_num_workers)

    with open(os.path.join(args.working_directory, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    algo.shutdown(shutdown_workers=True)
    NS.shutdown()

else:
    time.sleep(30)

    # create a worker object on all cluster nodes with args.array_id != 1
    worker = Worker(host=host,
                    run_id=args.run_id,
                    working_directory=args.working_directory,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    wd=args.wd,
                    scheme=args.scheme,
                    warmstart_dir=args.warmstart_dir,
                    dataset=args.dataset,
                    debug=args.debug)

    # connect to master
    worker.load_nameserver_credentials(args.working_directory)
    worker.run(background=False)
    exit(0)
