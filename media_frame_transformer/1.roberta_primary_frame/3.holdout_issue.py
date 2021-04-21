import sys
from os.path import join

from config import ARCH, BATCHSIZE, FOLDS_TO_RUN, ISSUES, KFOLD, MODELS_DIR
from media_frame_transformer.dataset import (
    PrimaryFrameDataset,
    get_kfold_primary_frames_datasets,
)
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.text_samples import load_all_text_samples

_arch = sys.argv[1]
EXPERIMENT_NAME = f"13.{_arch}"


def _train():
    path2datasets = {}

    for holdout_issue in ISSUES:
        model_name = f"holdout_{holdout_issue}"
        # train on all issues other than the holdout one
        train_issues = [iss for iss in ISSUES if iss != holdout_issue]

        kfold_datasets = get_kfold_primary_frames_datasets(train_issues)
        holdout_issue_all_samples = load_all_text_samples(
            [holdout_issue], split="train", task="primary_frame"
        )
        holdout_issue_dataset = PrimaryFrameDataset(holdout_issue_all_samples)

        for ki in FOLDS_TO_RUN:
            path2datasets[
                join(MODELS_DIR, EXPERIMENT_NAME, model_name, f"fold_{ki}")
            ] = {
                "train": kfold_datasets[ki]["train"],
                "valid": kfold_datasets[ki]["valid"],
                "additional_valid_datasets": {"holdout_issue": holdout_issue_dataset},
            }
    run_experiments(_arch, path2datasets, batchsize=BATCHSIZE)


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
