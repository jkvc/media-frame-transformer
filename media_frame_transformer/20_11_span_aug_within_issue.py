from os import mkdir
from os.path import exists, join

import pandas as pd
from config import ISSUES, MODELS_DIR

from media_frame_transformer import models
from media_frame_transformer.data_aug import (
    augment_train_splits,
    get_kfold_span_frame_train_samples,
)
from media_frame_transformer.dataset import (
    fold2split2samples_to_datasets,
    load_kfold_primary_frame_samples,
)
from media_frame_transformer.learning import get_kfold_metrics, train
from media_frame_transformer.utils import mkdir_overwrite, write_str_list_as_txt

EXPERIMENT_NAME = "2.0.1.1.roberta_meddrop_half.zeroth.w2.min150"
ARCH = "roberta_meddrop"

AUG_WEIGHT = 0.2

ZEROTH_FOLD_ONLY = False
KFOLD = 8

BATCHSIZE = 25


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
        aug_fold2samples = get_kfold_span_frame_train_samples(
            [issue], KFOLD, AUG_WEIGHT
        )
        augment_train_splits(fold2split2samples, aug_fold2samples)
        print(">>  after aug", len(fold2split2samples[0]["train"]))

        augmented_datasets = fold2split2samples_to_datasets(fold2split2samples)
        for ki, datasets in enumerate(augmented_datasets):

            if ZEROTH_FOLD_ONLY and ki != 0:
                break

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
                BATCHSIZE,
            )

            # mark done
            write_str_list_as_txt(["."], join(save_fold_path, "_complete"))


def _valid():
    root_path = join(MODELS_DIR, EXPERIMENT_NAME)
    assert exists(
        root_path
    ), f"{root_path} does not exist, choose the correct experiment name"

    metrics_save_filepath = join(root_path, "metrics.csv")
    assert not exists(metrics_save_filepath)

    issue2metrics = {}
    for issue in ISSUES:
        print(issue)
        issue_path = join(root_path, issue)

        metrics = get_kfold_metrics(
            [issue],
            KFOLD,
            issue_path,
            valid_on_train_also=False,
            zeroth_fold_only=ZEROTH_FOLD_ONLY,
        )
        issue2metrics[issue] = metrics

    df = pd.DataFrame.from_dict(issue2metrics, orient="index")
    df.loc["mean"] = df.mean()
    print(df)
    df.to_csv(metrics_save_filepath)


if __name__ == "__main__":
    _train()
    _valid()
