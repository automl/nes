import os
import torch
import warnings
warnings.filterwarnings('ignore')

from ConfigSpace.read_and_write import json as config_space_json_r_w
from hpbandster.core.worker import Worker
from hpbandster.core.result import logged_results_to_HBS_result as res_loader

from nes.ensemble_selection.create_baselearners import create_baselearner
from nes.optimizers.baselearner_train.utils import parse_config
from nes.optimizers.baselearner_train.train import run_train
from nes.optimizers.baselearner_train.genotypes import Genotype


class DARTSWorker(Worker):
    def __init__(self, working_directory, num_epochs, batch_size, *args,
                 scheme='nes_re', dataset='fmnist', warmstart_dir=None,
                 debug=False, n_workers=4, only_predict=False,
                 lr=0.025, wd=3e-4, drop_prob=0.3, anchor=False, anch_coeff=1,
                 full_train=False, n_datapoints=None,
                 oneshot=False, saved_model=None, **kwargs):
        """
        Args:
            working_directory (str): directory where results are written
            num_epochs        (int): number of total epochs to train the baselearner
            batch_size        (int): mini-batch size during training
            scheme            (str): scheme name
            dataset           (str): dataset name
            warmstart_dir     (str): directory where previous results are stored.
                Used only when it is not None
            debug            (bool): run the train only for one mini-batch when True
            n_workers         (int): number of workers in pytorch dataloader
            only_predict     (bool): run only inference for the saved models
            lr              (float): learning rate to train models
            wd              (float): weight decay value during model training
            drop_prob       (float): scheduled drop path max probability
            anchor           (bool): enable or not anchored ensembles' training
            anch_coeff      (float): anchor coefficient
            full_train       (bool): set to True if using all the examples in
                the training set.
            n_datapoints      (int): number of datapoints used for training
            oneshot          (bool): if True use the predictions coming from
                the weight sharing model.
            saved_model       (str): path to the oneshot saved model weights

        Returns:
            None

        """
        self.working_directory = working_directory
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.drop_prob = drop_prob
        self.debug = debug
        self.scheme = scheme
        self.warmstart_dir = warmstart_dir
        self.dataset = dataset
        self.n_workers = n_workers
        self.data_path = 'data/tiny-imagenet-200/' if dataset == 'tiny' else 'data/'
        self.only_predict = only_predict
        self.oneshot = oneshot
        self.anchor = anchor
        self.anch_coeff = anch_coeff
        self.full_train = full_train
        self.saved_model = saved_model
        self.n_datapoints = n_datapoints
        super().__init__(*args, **kwargs)

    def compute(self, config, budget, config_id, **kwargs):
        """Method that trains a given architecture, wraps it with a Baselearner
            object and saves it.

        Args:
            config     (dict): dictionary containing the configurations
                (architecture) that needs to be trained
            budget    (float): amount of epochs the model can use to train
            config_id (tuple): a 3-tuple as follows (model_id, 0, 0). Only the
                first element is used.

        Returns:
            dict with mandatory fields:
                'loss' (float)
                'info' (dict)
        """
        device = torch.device(f'cuda:0')
        model_id = int(config_id[0])

        # when not DeepEns train with the just one seed_id = model_id
        if 'seed_id' in kwargs:
            seed_id = kwargs['seed_id']
        else:
            seed_id = model_id

        # load the previous results if warmstarting
        if self.warmstart_dir is not None:
            previous_res = res_loader(self.warmstart_dir)
            model_id += len(previous_res.get_all_runs())

        # directory where to write the training results
        dest_dir = os.path.join(self.working_directory, 'run_'+str(model_id))

        if not isinstance(config, Genotype):
            # hpbandster passes a configspace object instead
            genotype = parse_config(config, self.get_configspace())
        else:
            genotype = config

        if not self.only_predict:
            # compute the baselearner prediction and return
            run_train(seed=seed_id,
                      arch_id=model_id,
                      arch=str(genotype),
                      num_epochs=self.num_epochs,
                      bslrn_batch_size=self.batch_size,
                      exp_name=dest_dir,
                      logger=self.logger,
                      data_path=self.data_path,
                      mode='train',
                      dataset=self.dataset,
                      debug=self.debug,
                      anchor=self.anchor,
                      anch_coeff=self.anch_coeff,
                      n_workers=self.n_workers,
                      lr=self.lr,
                      wd=self.wd,
                      drop_prob=self.drop_prob,
                      full_train=self.full_train,
                      n_datapoints=self.n_datapoints,
                      **kwargs)

        if self.oneshot:
            model_ckpt = self.saved_model
        else:
            model_ckpt = os.path.join(dest_dir,
                                      f'arch_{model_id}_init_{seed_id}_epoch_{self.num_epochs}.pt')

        # create a nes.ensemble_selection.containers.Baselearner object from
        # the trained architecture and compute the predictions for different
        # severity levels 
        baselearner = create_baselearner(state_dict_dir=model_ckpt,
                                         genotype=genotype,
                                         arch_seed=model_id,
                                         init_seed=seed_id,
                                         scheme=self.scheme,
                                         dataset=self.dataset,
                                         device=device,
                                         save_dir=dest_dir,
                                         n_datapoints=self.n_datapoints,
                                         nb201=False,
                                         oneshot=self.oneshot,
                                         **kwargs)

        print('BASLEARNER data:')
        print('LR: {:.6f}, WD: {:.6f}, anch: {:.6f}'.format(self.lr, self.wd, self.anch_coeff))

        return ({
            # this is a mandatory field to run hpbandster. Not used by NES-RE
            # though directly
            'loss': baselearner.evals['val']['0']['loss'],
            # can be used for any user-defined information - also mandatory
            'info': {'dest_dir': dest_dir,
                     'model_id': model_id}
        })

    @staticmethod
    def get_configspace():
        """Returns the ConfigSpace object for the search space

        Args:
            None

        Returns:
            ConfigSpace.ConfigurationSpace: a ConfigSpace object
        """
        with open(os.path.join('nes/darts/baselearner_train/space_encoding/configspace.json'),
                  'r') as fh:
            json_string = fh.read()
            config_space = config_space_json_r_w.read(json_string)
        return config_space


