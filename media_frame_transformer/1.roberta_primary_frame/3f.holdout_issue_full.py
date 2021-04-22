import sys
from collections import defaultdict
from os import makedirs
from os.path import join
from pprint import pprint
from random import Random

import media_frame_transformer.models_roberta  # noqa
import numpy as np
import torch
from config import BATCHSIZE, DATASET_SIZES, ISSUES, MODELS_DIR
from media_frame_transformer.dataset import (
    PrimaryFrameDataset,
    calculate_primary_frame_labelprops,
)
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.learning import (
    N_DATALOADER_WORKER,
    VALID_BATCHSIZE,
    valid_epoch,
)
from media_frame_transformer.text_samples import load_all_text_samples
from media_frame_transformer.utils import DEVICE, save_json
from torch.utils.data import DataLoader

_arch = sys.argv[1]
EXPERIMENT_NAME = f"3f.{_arch}"


RNG = Random()
RNG_SEED = 0xDEADBEEF
RNG.seed(RNG_SEED)

NTRIALS_EVAL_UNSEEN_ESTIMATED_DISTRIBUTION = 5


def _train():
    path2datasets = {}

    for holdout_issue in ISSUES:
        model_name = f"holdout_{holdout_issue}"

        train_issues = [iss for iss in ISSUES if iss != holdout_issue]
        train_issues_all_samples = load_all_text_samples(
            train_issues,
            split="train",
            task="primary_frame",
        )
        train_issue_dataset = PrimaryFrameDataset(train_issues_all_samples)

        holdout_issue_all_samples = load_all_text_samples(
            [holdout_issue],
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
        num_early_stop_non_improve_epoch=7,
        max_epochs=10,
    )


def _eval_unseen_estimated_distribution():
    issue2samplesize2trial2f1 = defaultdict(lambda: defaultdict(dict))
    for issue in ISSUES:
        model = torch.load(
            join(MODELS_DIR, EXPERIMENT_NAME, f"holdout_{issue}", "checkpoint.pth")
        ).to(DEVICE)

        all_holdout_issue_samples = load_all_text_samples(
            [issue],
            split="train",
            task="primary_frame",
        )
        dataset = PrimaryFrameDataset(all_holdout_issue_samples)

        for trial in range(NTRIALS_EVAL_UNSEEN_ESTIMATED_DISTRIBUTION):
            RNG.shuffle(all_holdout_issue_samples)

            for numsample in DATASET_SIZES:
                selected_samples = all_holdout_issue_samples[:numsample]
                estimated_issue2labelprops = calculate_primary_frame_labelprops(
                    selected_samples
                )
                dataset.issue2labelprops = estimated_issue2labelprops
                loader = DataLoader(
                    dataset, batch_size=VALID_BATCHSIZE, num_workers=N_DATALOADER_WORKER
                )
                metrics = valid_epoch(model, loader)
                issue2samplesize2trial2f1[issue][numsample][trial] = metrics["f1"]

            pprint(issue2samplesize2trial2f1)

    save_json(
        dict(issue2samplesize2trial2f1),
        join(
            MODELS_DIR,
            EXPERIMENT_NAME,
            "estimated_distribution_f1.json",
        ),
    )


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))

    if _arch.endswith("+dev"):
        _eval_unseen_estimated_distribution()
