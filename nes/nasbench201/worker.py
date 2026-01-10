import os
import torch
import warnings
warnings.filterwarnings('ignore')

from hpbandster.core.worker import Worker
from nes.ensemble_selection.create_baselearners import create_baselearner


class NB201Worker(Worker):
    def __init__(self, working_directory, *args, scheme='nes_re',
                 dataset='cifar10', n_datapoints=None, load_preds=True, **kwargs):
        """
        Args:
            working_directory (str): directory where results are written
            scheme            (str): scheme name
            dataset           (str): dataset name
            n_datapoints      (int): determines the number of data used in train and val
                                     splits
        Returns:
            None

        """
        self.working_directory = working_directory
        self.scheme = scheme
        self.dataset = dataset
        self.n_datapoints = n_datapoints
        self.load_preds = load_preds
        super().__init__(*args, **kwargs)

    def compute(self, config, budget, config_id, seed_id, **kwargs):
        """Method that trains a given architecture, wraps it with a Baselearner
            object and saves it.

        Args:
            config     (dict): dictionary containing the configurations
                (architecture) that needs to be trained
            budget    (float): amount of epochs the model can use to train
            config_id   (int): arch_id
            seed_id     (int): seed_id

        Returns:
            dict with mandatory fields:
                'loss' (float)
                'info' (dict)
        """
        device = torch.device(f'cpu')#cuda:0
        arch_id = config_id

        # directory where to write the training results
        dest_dir = os.path.join(self.working_directory, 'run_'+str(arch_id))

        if self.load_preds:
            seed_list = list(range(3))
            if seed_id == 0: seed = 0
            elif seed_id == 1: seed = 1
            else: seed = 2

            model_ckpt =\
                    f"data/nb201_preds/{self.dataset}/{seed}/{config}.pt"
            #NOTE: nb201 does not have all predictions stored for each seed, so
            # we have to pick another seed if that case happens
            if not os.path.exists(model_ckpt):
                seed_list.remove(seed)
                model_ckpt =\
                        f"data/nb201_preds/{self.dataset}/{seed_list[0]}/{config}.pt"
                if not os.path.exists(model_ckpt):
                    model_ckpt =\
                        f"data/nb201_preds/{self.dataset}/{seed_list[1]}/{config}.pt"
        else:
            model_ckpt =\
                f'/data/aad/image_datasets/nb201_new/NAS-BENCH-102-4-v1.0-archive/{config}-FULL.pth'

        # create a nes.ensemble_selection.containers.Baselearner object from
        # the trained architecture and compute the predictions for different
        # severity levels 
        baselearner = create_baselearner(state_dict_dir=model_ckpt,
                                         genotype=None,
                                         arch_seed=arch_id,
                                         init_seed=seed_id,
                                         scheme=self.scheme,
                                         dataset=self.dataset,
                                         device=device,
                                         save_dir=dest_dir,
                                         n_datapoints=self.n_datapoints,
                                         nb201=True,
                                         oneshot=False,
                                         load_preds=self.load_preds,
                                         **kwargs)

        return ({
            # this is a mandatory field to run hpbandster. Not used by NES-RE
            # though directly
            'loss': baselearner.evals['val']['0']['loss'],
            # can be used for any user-defined information - also mandatory
            'info': {'dest_dir': dest_dir,
                     'model_id': arch_id}
        })


