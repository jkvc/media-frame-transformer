import sys
from collections import defaultdict
from os import makedirs
from os.path import join
from pprint import pprint
from random import Random

import media_frame_transformer.models_roberta  # noqa
from config import BATCHSIZE, ISSUES, MODELS_DIR
from media_frame_transformer.dataset import PrimaryFrameDataset
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.text_samples import load_all_text_samples

_arch = sys.argv[1]
EXPERIMENT_NAME = f"6f.{_arch}"


def _train():
    path2datasets = {}

    for issue in ISSUES:
        model_name = f"holdout_{issue}"

        train_issues_all_samples = load_all_text_samples(
            [issue],
            split="train",
            task="primary_frame",
        )
        train_issue_dataset = PrimaryFrameDataset(train_issues_all_samples)

        valid_issues = [i for i in ISSUES if i != issue]
        holdout_issue_all_samples = load_all_text_samples(
            valid_issues,
            split="train",
            task="primary_frame",
        )
        holdout_issue_dataset = PrimaryFrameDataset(holdout_issue_all_samples)

        path2datasets[join(MODELS_DIR, EXPERIMENT_NAME, model_name)] = {
            "train": train_issue_dataset,
            "valid": holdout_issue_dataset,
        }
    run_experiments(
        _arch,
        path2datasets,
        batchsize=BATCHSIZE,
        num_early_stop_non_improve_epoch=4,
    )


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
