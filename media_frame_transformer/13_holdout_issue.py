from os import mkdir
from os.path import exists, join

from config import ISSUES, MODELS_DIR

from media_frame_transformer import models
from media_frame_transformer.dataset import (
    PrimaryFrameDataset,
    get_kfold_primary_frames_datasets,
    load_all_primary_frame_samples,
)
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiment_config import (
    ARCH,
    BATCHSIZE,
    FOLDS_TO_RUN,
    KFOLD,
)
from media_frame_transformer.learning import train
from media_frame_transformer.utils import mkdir_overwrite, write_str_list_as_txt

EXPERIMENT_NAME = f"1.3.{ARCH}"


def _train():
    save_root = join(MODELS_DIR, EXPERIMENT_NAME)
    if not exists(save_root):
        mkdir(save_root)

    for holdout_issue in ISSUES:
        model_name = f"holdout_{holdout_issue}"
        print(model_name)
        save_issue_path = join(save_root, model_name)
        if not exists(save_issue_path):
            mkdir(save_issue_path)

        # train on all issues other than the holdout one
        train_issues = [iss for iss in ISSUES if iss != holdout_issue]

        kfold_datasets = get_kfold_primary_frames_datasets(train_issues, KFOLD)
        holdout_issue_all_samples = load_all_primary_frame_samples([holdout_issue])
        holdout_issue_dataset = PrimaryFrameDataset(holdout_issue_all_samples)

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
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                logdir=save_fold_path,
                additional_valid_datasets={"holdout_issue": holdout_issue_dataset},
                batchsize=BATCHSIZE,
            )

            # mark done
            write_str_list_as_txt(["."], join(save_fold_path, "_complete"))


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
