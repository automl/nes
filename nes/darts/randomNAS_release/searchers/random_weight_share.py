import sys
import os
import shutil
import logging
import inspect
import pickle
import argparse
import numpy as np

from nes.randomNAS_release.benchmarks.cnn.darts.darts_wrapper_discrete import DartsWrapper

class Rung:
    def __init__(self, rung, nodes):
        self.parents = set()
        self.children = set()
        self.rung = rung
        for node in nodes:
            n = nodes[node]
            if n.rung == self.rung:
                self.parents.add(n.parent)
                self.children.add(n.node_id)

class Node:
    def __init__(self, parent, arch, node_id, rung):
        self.parent = parent
        self.arch = arch
        self.node_id = node_id
        self.rung = rung
    def to_dict(self):
        out = {'parent':self.parent, 'arch': self.arch, 'node_id': self.node_id, 'rung': self.rung}
        if hasattr(self, 'objective_val'):
            out['objective_val'] = self.objective_val
        return out

class Random_NAS:
    def __init__(self, B, model, seed, save_dir):
        self.save_dir = save_dir

        self.B = B
        self.model = model
        self.seed = seed

        self.iters = 0

        self.arms = {}
        self.node_id = 0

    def print_summary(self):
        logging.info(self.parents)
        objective_vals = [(n,self.arms[n].objective_val) for n in self.arms if hasattr(self.arms[n],'objective_val')]
        objective_vals = sorted(objective_vals,key=lambda x:x[1])
        best_arm = self.arms[objective_vals[0][0]]
        val_ppl = self.model.evaluate(best_arm.arch, split='valid')
        logging.info(objective_vals)
        logging.info('best valid ppl: %.2f' % val_ppl)


    def get_arch(self):
        arch = self.model.sample_arch()
        self.arms[self.node_id] = Node(self.node_id, arch, self.node_id, 0)
        self.node_id += 1
        return arch

    def save(self):
        to_save = {a: self.arms[a].to_dict() for a in self.arms}
        # Only replace file if save successful so don't lose results of last pickle save
        with open(os.path.join(self.save_dir,'results_tmp.pkl'),'wb') as f:
            pickle.dump(to_save, f)
        shutil.copyfile(os.path.join(self.save_dir, 'results_tmp.pkl'), os.path.join(self.save_dir, 'results.pkl'))

        self.model.save()

    def run(self):
        while self.iters < self.B:
            arch = self.get_arch()
            self.model.train_batch(arch)
            self.iters += 1
            if self.iters % 500 == 0:
                self.save()
        self.save()

    def get_eval_arch(self, rounds=None):
        #n_rounds = int(self.B / 7 / 1000)
        if rounds is None:
            n_rounds = max(1,int(self.B/10000))
        else:
            n_rounds = rounds
        best_rounds = []
        for r in range(n_rounds):
            sample_vals = []
            for _ in range(1000):
                arch = self.model.sample_arch()
                try:
                    ppl = self.model.evaluate(arch)
                except Exception as e:
                    ppl = 1000000
                logging.info(arch)
                logging.info('objective_val: %.3f' % ppl)
                sample_vals.append((arch, ppl))
            sample_vals = sorted(sample_vals, key=lambda x:x[1])

            full_vals = []
            if 'split' in inspect.getargspec(self.model.evaluate).args:
                for i in range(10):
                    arch = sample_vals[i][0]
                    try:
                        ppl = self.model.evaluate(arch, split='valid')
                    except Exception as e:
                        ppl = 1000000
                    full_vals.append((arch, ppl))
                full_vals = sorted(full_vals, key=lambda x:x[1])
                logging.info('best arch: %s, best arch valid performance: %.3f' % (' '.join([str(i) for i in full_vals[0][0]]), full_vals[0][1]))
                best_rounds.append(full_vals[0])
            else:
                best_rounds.append(sample_vals[0])
        return best_rounds

def main(args):
    # Fill in with root output path
    root_dir = '/home/zelaa/NIPS20/robust_ensembles/nes/randomNAS_release/results'
    if args.save_dir is None:
        save_dir = os.path.join(root_dir, '%s/random/trial%d' % (args.benchmark, args.seed))
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.eval_only:
        assert args.save_dir is not None

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(args)

    data_size = 25000
    time_steps = 1
    B = int(args.epochs * data_size / args.batch_size / time_steps)
    model = DartsWrapper(save_dir, args.seed, args.batch_size, args.grad_clip, args.epochs, init_channels=args.init_channels)

    searcher = Random_NAS(B, model, args.seed, save_dir)
    logging.info('budget: %d' % (searcher.B))
    if not args.eval_only:
        searcher.run()
        #archs = searcher.get_eval_arch()
    else:
        np.random.seed(args.seed+1)
        archs = searcher.get_eval_arch(2)
    logging.info(archs)
    arch = ' '.join([str(a) for a in archs[0][0]])
    with open('/home/zelaa/NIPS20/robust_ensembles/nes/randomNAS_release/arch','w') as f:
        f.write(arch)
    return arch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for SHA with weight sharing')
    parser.add_argument('--benchmark', dest='benchmark', type=str, default='cnn')
    parser.add_argument('--seed', dest='seed', type=int, default=100)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=0.25)
    parser.add_argument('--save_dir', dest='save_dir', type=str, default=None)
    parser.add_argument('--eval_only', dest='eval_only', type=int, default=0)
    # PTB only argument. config=search uses proxy network for shared weights while
    # config=eval uses proxyless network for shared weights.
    parser.add_argument('--config', dest='config', type=str, default="search")
    # CIFAR-10 only argument.  Use either 16 or 24 for the settings for random search
    # with weight-sharing used in our experiments.
    parser.add_argument('--init_channels', dest='init_channels', type=int, default=16)
    args = parser.parse_args()

    main(args)







