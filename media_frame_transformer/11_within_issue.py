from collections import defaultdict
from os import mkdir, write
from os.path import exists, join

import pandas as pd
from config import ISSUES, MODELS_DIR

from media_frame_transformer import models
from media_frame_transformer.dataset import get_kfold_primary_frames_datasets
from media_frame_transformer.learning import get_kfold_metrics, train
from media_frame_transformer.utils import mkdir_overwrite, write_str_list_as_txt

EXPERIMENT_NAME = "1.1.roberta_meddrop_half.zeroth"
ARCH = "roberta_meddrop_half"


KFOLD = 8
ZEROTH_FOLD_ONLY = True
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

        kfold_datasets = get_kfold_primary_frames_datasets([issue], KFOLD)
        for ki, datasets in enumerate(kfold_datasets):
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
