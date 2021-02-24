from os import mkdir
from os.path import exists, join

import pandas as pd
from config import ISSUES, MODELS_DIR

from media_frame_transformer import models
from media_frame_transformer.data_aug import (
    augment_train_splits,
    get_kfold_multi_span_frame_train_samples_predefined_issue,
    get_kfold_single_span_frame_train_samples,
)
from media_frame_transformer.dataset import (
    fold2split2samples_to_datasets,
    load_kfold_primary_frame_samples,
)
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.learning import get_kfold_metrics, train
from media_frame_transformer.utils import mkdir_overwrite, write_str_list_as_txt

EXPERIMENT_NAME = "2.1.1.1.meddrop_half.w2.2x"
ARCH = "roberta_meddrop_half"

AUG_WEIGHT = 0.2
AUG_SET_SIZE_MULTIPLIER = 2

KFOLD = 8
FOLDS_TO_RUN = [0, 1, 2]

BATCHSIZE = 50


def _train():
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    if not exists(save_root):
        mkdir(save_root)

    for issue in ISSUES:
        print(issue)
        save_issue_path = join(save_root, issue)
        if not exists(save_issue_path):
            mkdir(save_issue_path)

        fold2split2samples = load_kfold_primary_frame_samples([issue], KFOLD)
        print(">> before aug", len(fold2split2samples[0]["train"]))

        aug_fold2samples = get_kfold_multi_span_frame_train_samples_predefined_issue(
            issue, KFOLD, AUG_SET_SIZE_MULTIPLIER, AUG_WEIGHT
        )
        augment_train_splits(fold2split2samples, aug_fold2samples)
        print(">>  after aug", len(fold2split2samples[0]["train"]))

        augmented_datasets = fold2split2samples_to_datasets(fold2split2samples)
        for ki, datasets in enumerate(augmented_datasets):

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
                logdir=save_fold_path,
                batchsize=BATCHSIZE,
            )

            # mark done
            write_str_list_as_txt(["."], join(save_fold_path, "_complete"))


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
