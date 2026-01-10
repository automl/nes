import os
import torch
import argparse
from nes.ensemble_selection.utils import get_dataloaders_and_classes
from nes.ensemble_selection.create_baselearners import create_baselearner

models_dir = '/data/aad/image_datasets/nb201_new/NAS-BENCH-102-4-v1.0-archive/'
device = torch.device(f'cuda:0')

def main(seed, dataset):
    missing_archs = list()
    dataloaders, num_classes = get_dataloaders_and_classes(dataset, device,
                                                           True, None,
                                                           range(1))
    for i, a in enumerate(os.listdir(models_dir)):
        print(i, a)
        arch_id = seed_id = i
        model_id = a[:6]
        dest_dir = os.path.join('data/nb201_preds_c10', dataset, str(seed))
        model_ckpt = models_dir + a
        try:
            baselearner = create_baselearner(state_dict_dir=model_ckpt, genotype=None,
                                             arch_seed=arch_id, init_seed=seed,
                                             scheme='nes-re', dataset=dataset, device=device,
                                             save_dir=dest_dir, n_datapoints=None,
                                             nb201=True, oneshot=False,
                                             ignore_missing_seed=False,
                                             save_evals=False,
                                             load_preds=False, only_clean=True,
                                             dataloader=dataloaders,
                                             num_classes=num_classes)
        except KeyError:
            print(f'Arch with id {model_id} missing')
            missing_archs.append(model_id)
            continue

        # save the predictions and evaluations
        baselearner.save_evals(dest_dir, model_id, force_overwrite=True)

    torch.save(missing_archs, os.path.join('data/nb201_preds_c10', dataset,
                                           f'missing_archs_{seed}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="cifar10")

    args = parser.parse_args()
    main(args.seed, args.dataset)
