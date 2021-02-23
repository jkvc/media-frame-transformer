from os import mkdir
from os.path import exists, join

import numpy as np
import pandas as pd
from config import ISSUES, MODELS_DIR

from media_frame_transformer import models
from media_frame_transformer.dataset import (
    fold2split2samples_to_datasets,
    get_kfold_primary_frames_datasets,
    load_kfold_primary_frame_samples,
)
from media_frame_transformer.learning import get_kfold_metrics, train
from media_frame_transformer.utils import mkdir_overwrite, write_str_list_as_txt

EXPERIMENT_NAME = "3.0.1.1.meddrop_half"
ARCH = "roberta_meddrop_half"


KFOLD = 8
FOLDS_TO_RUN = [0, 1, 2, 3]

BATCHSIZE = 50
DATASET_SIZE_PROPS = [0.2, 0.4, 0.6, 0.8, 1.0]


def _train():
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    if not exists(save_root):
        mkdir(save_root)

    # root/prop/issue/fold
    for prop in DATASET_SIZE_PROPS:
        save_prop_path = join(save_root, str(prop))
        if not exists(save_prop_path):
            mkdir(save_prop_path)

        for issue in ISSUES:
            print(issue)
            save_issue_path = join(save_prop_path, issue)
            if not exists(save_issue_path):
                mkdir(save_issue_path)

            # kfold_datasets = get_kfold_primary_frames_datasets([issue], KFOLD)
            fold2split2samples = load_kfold_primary_frame_samples([issue], KFOLD)

            # subsample all folds
            subsampled_fold2split2samples = []
            for ki, split2samples in enumerate(fold2split2samples):
                num_train_samples = int(np.ceil(len(split2samples["train"]) * prop))
                print(">> prop", prop, "split", ki, "num train", num_train_samples)
                subsampled_fold2split2samples.append(
                    {
                        "valid": split2samples["valid"],
                        "train": split2samples["train"][:num_train_samples],
                    }
                )
            kfold_datasets = fold2split2samples_to_datasets(
                subsampled_fold2split2samples
            )

            for ki, datasets in enumerate(kfold_datasets):
                if ki not in FOLDS_TO_RUN:
                    print(">> not running fold", ki)
                    continue

                # skip done
                save_fold_path = join(save_issue_path, f"fold_{ki}")
                if exists(join(save_fold_path, "_complete")):
                    print(">> skip", ki)
                    continue
                mkdir_overwrite(save_fold_path)

                train_dataset = datasets["train"]
                valid_dataset = datasets["valid"]

                model = models.get_model(ARCH)
                train(
                    model,
                    train_dataset,
                    valid_dataset,
                    save_fold_path,
                    max_epochs=30,
                    batchsize=BATCHSIZE,
                )

                # mark done
                write_str_list_as_txt(["."], join(save_fold_path, "_complete"))


if __name__ == "__main__":
    _train()
