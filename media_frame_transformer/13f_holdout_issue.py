import sys
from os.path import join

from config import ISSUES, MODELS_DIR

from media_frame_transformer.dataset import (
    PrimaryFrameDataset,
    get_kfold_primary_frames_datasets,
    load_all_primary_frame_samples,
)
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiment_config import BATCHSIZE, FOLDS_TO_RUN, KFOLD
from media_frame_transformer.experiments import run_experiments

_arch = sys.argv[1]
EXPERIMENT_NAME = f"13f.{_arch}"


def _train():
    path2datasets = {}

    for holdout_issue in ISSUES:
        model_name = f"holdout_{holdout_issue}"

        train_issues = [iss for iss in ISSUES if iss != holdout_issue]
        train_issues_all_samples = load_all_primary_frame_samples(train_issues)
        train_issue_dataset = PrimaryFrameDataset(train_issues_all_samples)

        holdout_issue_all_samples = load_all_primary_frame_samples([holdout_issue])
        holdout_issue_dataset = PrimaryFrameDataset(holdout_issue_all_samples)

        path2datasets[join(MODELS_DIR, EXPERIMENT_NAME, model_name)] = {
            "train": train_issue_dataset,
            "valid": holdout_issue_dataset,
        }
    run_experiments(_arch, path2datasets, batchsize=BATCHSIZE)


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
