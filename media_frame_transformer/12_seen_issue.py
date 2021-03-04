from os import mkdir
from os.path import exists, join

import pandas as pd
import torch
import torch.nn as nn
from config import ISSUES, MODELS_DIR

from media_frame_transformer import models
from media_frame_transformer.dataset import get_kfold_primary_frames_datasets
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiment_config import (
    ARCH,
    BATCHSIZE,
    FOLDS_TO_RUN,
    KFOLD,
)
from media_frame_transformer.learning import get_kfold_metrics, train
from media_frame_transformer.utils import mkdir_overwrite, write_str_list_as_txt

EXPERIMENT_NAME = f"1.2.{ARCH}"


def _train():
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    if not exists(save_root):
        mkdir(save_root)

    kfold_datasets = get_kfold_primary_frames_datasets(ISSUES, KFOLD)
    for ki, datasets in enumerate(kfold_datasets):
        if ki not in FOLDS_TO_RUN:
            print(">> not running fold", ki)
            continue

        save_fold = join(save_root, f"fold_{ki}")
        if exists(join(save_fold, "_complete")):
            print(">> skip", ki)
            continue
        mkdir_overwrite(save_fold)

        train_dataset = datasets["train"]
        valid_dataset = datasets["valid"]

        model = models.get_model(ARCH)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        train(
            model,
            train_dataset,
            valid_dataset,
            save_fold,
            BATCHSIZE,
        )

        write_str_list_as_txt(["."], join(save_fold, "_complete"))


# def _valid_combined_issue():
#     experiment_root_path = join(MODELS_DIR, EXPERIMENT_NAME)
#     assert exists(
#         experiment_root_path
#     ), f"{experiment_root_path} does not exist, choose the correct experiment name"

#     metrics_save_filepath = join(experiment_root_path, "metrics_combined_issues.csv")
#     assert not exists(metrics_save_filepath)

#     metrics = get_kfold_metrics(
#         ISSUES,
#         KFOLD,
#         experiment_root_path,
#         zeroth_fold_only=ZEROTH_FOLD_ONLY,
#     )
#     metrics = {"all": metrics}

#     df = pd.DataFrame.from_dict(metrics, orient="index")
#     print(df)
#     df.to_csv(metrics_save_filepath)


# def _valid_individual_issue():
#     experiment_root_path = join(MODELS_DIR, EXPERIMENT_NAME)
#     assert exists(
#         experiment_root_path
#     ), f"{experiment_root_path} does not exist, choose the correct experiment name"

#     metrics_save_filepath = join(experiment_root_path, "metrics_individual_issues.csv")
#     assert not exists(metrics_save_filepath)

#     issue2metrics = {}
#     for issue in ISSUES:
#         print(issue)
#         metrics = get_kfold_metrics(
#             [issue],
#             KFOLD,
#             experiment_root_path,
#             zeroth_fold_only=ZEROTH_FOLD_ONLY,
#         )
#         issue2metrics[issue] = metrics

#     df = pd.DataFrame.from_dict(issue2metrics, orient="index")
#     df.loc["mean"] = df.mean()
#     print(df)
#     df.to_csv(metrics_save_filepath)


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
    # _valid_combined_issue()
    # _valid_individual_issue()
