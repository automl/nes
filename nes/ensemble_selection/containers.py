import os
import torch
import numpy as np

from torch.utils.data import TensorDataset
from collections import defaultdict
from types import SimpleNamespace
from pathlib import Path

from nes.ensemble_selection.utils import (
    make_predictions,
    evaluate_predictions,
    form_ensemble_pred,
    model_seeds,
)


METRICS = SimpleNamespace(loss="loss", accuracy="acc", error="error", ece="ece")


def check_to_avoid_overwrite(thing_to_check):
    """Makes a decorator which makes sure 'thing_to_check' is None before running func."""

    def _decorator(func):
        def careful_func(*args, force_overwrite=False, **kwargs):
            if force_overwrite:
                func(*args, **kwargs)
            else:
                if getattr(args[0], thing_to_check) is not None:
                    print(
                        f"Did not run to avoid overwriting. Set force_overwrite=True to overwrite."
                    )
                    return None
                else:
                    func(*args, **kwargs)

        return careful_func

    return _decorator


class Baselearner:
    """
    A container class for baselearner networks which can hold the nn.Module, 
    predictions (as tensors) and evaluations. It has methods for computing the predictions
    and evaluations on validation and test sets with shifts of varying severities. 
    """

    _cpu_device = torch.device("cpu")

    def __init__(
        self,
        model_id,
        severities,
        device=None,
        nn_module=None,
        preds=None,
        evals=None,
        model_config=None,
    ):
        self.model_id = model_id
        self.device = device
        self.nn_module = nn_module
        self.preds = preds
        self.evals = evals
        self.lsm_applied = False
        self.model_config = (
            model_config  # TODO: can probably be removed. do grep check to make sure.
        )
        self.severities = severities

    def to_device(self, device=None):
        if device is None:
            device = self._cpu_device

        if self.nn_module is not None:
            self.nn_module.to(device)

        if self.preds is not None:
            for key, dct in self.preds.items():
                for k, tsr_dst in dct.items():
                    dct[k] = TensorDataset(
                        tsr_dst.tensors[0].to(device), tsr_dst.tensors[1].to(device)
                    )

        self.device = device

    @check_to_avoid_overwrite("preds")
    def compute_preds(self, dataloaders, severities=None, num_classes=10):
        """
        Computes and stores the predictions of the model as tensors.

        Args:
            dataloaders (dict): Contains dataloaders for datasets over which to make predictions. See e.g.
                `create_dataloader_dict_fmnist` in `nes/ensemble_selection/utils.py`.
            severities (list-like, optional): Severity levels (as ints) of data shift.
            num_classes (int, optional): Number of classes.
        """
        if severities is None:
            severities = self.severities

        if dataloaders["metadata"]["device"] != self.device:
            raise ValueError(
                f'Data is on {dataloaders["metadata"]["device"]}, but baselearner is on {self.device}'
            )

        preds = defaultdict(dict)

        for severity in severities:
            for data_type in ["val", "test"]:
                loader = dataloaders[data_type][str(severity)]
                preds[data_type][str(severity)] = make_predictions(
                    self.nn_module, loader, self.device, num_classes=num_classes
                )

        self.preds = preds

    @check_to_avoid_overwrite("evals")
    def compute_evals(self, severities=None):
        if severities is None:
            severities = self.severities

        if self.preds is None:
            raise ValueError(
                "Baselearner predictions not available. Run .compute_preds(...) first."
            )

        evals = defaultdict(dict)

        for severity in severities:
            for data_type in ["val", "test"]:
                preds = self.preds[data_type][str(severity)]
                evaluation = evaluate_predictions(preds, self.lsm_applied)
                evaluation = {
                    METRICS.loss: evaluation["loss"],
                    METRICS.accuracy: evaluation["acc"],
                    METRICS.error: 1 - evaluation["acc"],
                    METRICS.ece: evaluation["ece"],
                }

                evals[data_type][str(severity)] = evaluation

        self.evals = evals

    def save(self, directory, force_overwrite=False):
        self.to_device(self._cpu_device)

        dir = os.path.join(
            directory,
            f"arch_{self.model_id.arch}_init_{self.model_id.init}_scheme_{self.model_id.scheme}",
        )

        if force_overwrite:
            Path(dir).mkdir(parents=True, exist_ok=True)
            print(f"Forcefully overwriting {dir}")
        else:
            Path(dir).mkdir(parents=True, exist_ok=False)

        torch.save(self.model_id._asdict(), os.path.join(dir, "model_id.pt"))

        if self.nn_module is not None:
            torch.save(self.nn_module, os.path.join(dir, "nn_module.pt"))

        torch.save(
            {"preds": self.preds, "evals": self.evals},
            os.path.join(dir, "preds_evals.pt"),
        )

    @classmethod
    def load(cls, dir, load_nn_module=False):

        model_id = model_seeds(**torch.load(os.path.join(dir, "model_id.pt")))

        preds, evals = torch.load(os.path.join(dir, "preds_evals.pt")).values()

        if load_nn_module:
            nn_module = torch.load(os.path.join(dir, "nn_module.pt"))
        else:
            nn_module = None

        # find number of severities
        assert len(preds["val"]) == len(
            preds["test"]
        )  # should be equal in current set up with CIFAR-10-C.
        num_sevs = len(preds["val"])
        severities = range(num_sevs)

        device = cls._cpu_device
        obj = cls(
            model_id=model_id,
            severities=severities,
            device=device,
            nn_module=nn_module,
            preds=preds,
            evals=evals,
        )

        return obj


def load_baselearner(model_id, load_nn_module, working_dir):
    dir = os.path.join(
        working_dir,
        f"arch_{model_id.arch}_init_{model_id.init}_scheme_{model_id.scheme}",
    )
    to_return = Baselearner.load(dir, load_nn_module)
    assert to_return.model_id == model_id
    return to_return


class Ensemble:
    """
    A container class for ensembles which holds Baselearner objects. It can hold
    and compute the ensemble's predictions, evaluations. 
    """

    def __init__(self, baselearners):
        if len(set(b.device for b in baselearners)) != 1:
            raise ValueError("All baselearners should be on the same device.")

        if len(set(b.severities for b in baselearners)) != 1:
            raise ValueError(
                "All baselearners should be evaluated on the same number of severities."
            )

        self.baselearners = baselearners
        self.ensemble_size = len(self.baselearners)
        self.lsm_applied = True

        self.preds = None
        self.evals = None

        self.avg_baselearner_evals = None

        self.oracle_preds = None
        self.oracle_evals = None

        self.device = self.baselearners[0].device
        self._cpu_device = torch.device("cpu")
        self.severities = self.baselearners[0].severities

    def to_device(self, device=None):
        if device is None:
            device = self._cpu_device

        for b in self.baselearners:
            b.to_device(device)

        if self.preds is not None:
            for key, dct in self.preds.items():
                for k, tsr_dst in dct.items():
                    dct[k] = TensorDataset(
                        tsr_dst.tensors[0].to(device), tsr_dst.tensors[1].to(device)
                    )

        self.device = device

    @check_to_avoid_overwrite("preds")
    def compute_preds(self, severities=None, combine_post_softmax=True):
        if severities is None:
            severities = self.severities
        ens_preds = defaultdict(dict)

        for severity in severities:
            for data_type in ["val", "test"]:
                preds_dict = {
                    b.model_id: b.preds[data_type][str(severity)].tensors[0]
                    for b in self.baselearners
                }
                labels = self.baselearners[0].preds[data_type][str(severity)].tensors[1]

                preds = form_ensemble_pred(
                    preds_dict,
                    lsm_applied=False,
                    combine_post_softmax=combine_post_softmax,
                )
                preds = TensorDataset(preds, labels)

                ens_preds[data_type][str(severity)] = preds

        self.preds = ens_preds

    @check_to_avoid_overwrite("evals")
    def compute_evals(self, severities=None):
        if severities is None:
            severities = self.severities

        if self.preds is None:
            raise ValueError(
                "Baselearners' predictions not available. Run .compute_preds(...) first."
            )

        evals = defaultdict(dict)

        for severity in severities:
            for data_type in ["val", "test"]:
                preds = self.preds[data_type][str(severity)]
                evaluation = evaluate_predictions(preds, self.lsm_applied)
                evaluation = {
                    METRICS.loss: evaluation["loss"],
                    METRICS.accuracy: evaluation["acc"],
                    METRICS.error: 1 - evaluation["acc"],
                    METRICS.ece: evaluation["ece"],
                }

                evals[data_type][str(severity)] = evaluation

        self.evals = evals

    @check_to_avoid_overwrite("avg_baselearner_evals")
    def compute_avg_baselearner_evals(self, severities=None):
        if severities is None:
            severities = self.severities

        evals = defaultdict(lambda: defaultdict(dict))

        for severity in severities:
            for data_type in ["val", "test"]:
                for metric in METRICS.__dict__.values():
                    avg = np.mean(
                        [
                            b.evals[data_type][str(severity)][metric]
                            for b in self.baselearners
                        ]
                    )
                    evals[data_type][str(severity)][metric] = avg

        self.avg_baselearner_evals = evals

    @check_to_avoid_overwrite("oracle_preds")
    def compute_oracle_preds(self, severities=None):
        if severities is None:
            severities = self.severities

        oracle_preds = defaultdict(dict)

        for severity in severities:
            for data_type in ["val", "test"]:
                labels = (
                    self.baselearners[0].preds[data_type][str(severity)].tensors[1]
                )  # labels of any baselearner are fine

                all_preds_list = [
                    b.preds[data_type][str(severity)].tensors[0].softmax(1)
                    for b in self.baselearners
                ]
                all_preds = torch.stack(all_preds_list, dim=1)

                _, oracle_selection = torch.max(
                    all_preds[range(len(all_preds)), :, labels], 1
                )
                _oracle_preds = all_preds[
                    range(len(all_preds)), oracle_selection, :
                ].log()

                oracle_preds[data_type][str(severity)] = TensorDataset(
                    _oracle_preds, labels
                )

        self.oracle_preds = oracle_preds

    @check_to_avoid_overwrite("oracle_evals")
    def compute_oracle_evals(self, severities=None):
        if severities is None:
            severities = self.severities
        if self.oracle_preds is None:
            raise ValueError(
                "Oracle's predictions not available. Run .compute_oracle_preds(...) first."
            )

        evals = defaultdict(dict)

        for severity in severities:
            for data_type in ["val", "test"]:
                preds = self.oracle_preds[data_type][str(severity)]
                evaluation = evaluate_predictions(preds, self.lsm_applied)
                evaluation = {
                    METRICS.loss: evaluation["loss"],
                    METRICS.accuracy: evaluation["acc"],
                    METRICS.error: 1 - evaluation["acc"],
                    METRICS.ece: evaluation["ece"],
                }

                evals[data_type][str(severity)] = evaluation

        self.oracle_evals = evals

