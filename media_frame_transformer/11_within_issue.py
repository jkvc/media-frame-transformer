from collections import defaultdict
from os import mkdir, write
from os.path import exists, join

import pandas as pd
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
from media_frame_transformer.utils import (
    mkdir_overwrite,
    save_json,
    write_str_list_as_txt,
)

EXPERIMENT_NAME = f"1.1.{ARCH}"


def _train():
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    if not exists(save_root):
        mkdir(save_root)

    for issue in ISSUES:
        print(issue)
        save_issue_path = join(save_root, issue)
        if not exists(save_issue_path):
            mkdir(save_issue_path)

        kfold_datasets = get_kfold_primary_frames_datasets([issue], KFOLD)
        for ki, datasets in enumerate(kfold_datasets):
            if ki not in FOLDS_TO_RUN:
                print(">> not running fold", ki)
                continue

            save_fold_path = join(save_issue_path, f"fold_{ki}")

            # skip done
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
                additional_valid_datasets={"test_valid": valid_dataset},
            )

            # mark done
            write_str_list_as_txt(["."], join(save_fold_path, "_complete"))


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
