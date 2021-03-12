import sys
from collections import defaultdict
from os import mkdir
from os.path import exists, join
from pprint import pprint
from random import Random, shuffle

import numpy as np
from config import ISSUES, MODELS_DIR
from matplotlib.pyplot import xlabel

from media_frame_transformer import models
from media_frame_transformer.dataset import (PrimaryFrameDataset,
                                             fold2split2samples_to_datasets,
                                             load_kfold_primary_frame_samples)
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiment_config import (ARCH, BATCHSIZE,
                                                       FOLDS_TO_RUN, KFOLD)
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.learning import train
from media_frame_transformer.utils import (load_json, mkdir_overwrite,
                                           write_str_list_as_txt)
from media_frame_transformer.viualization import (
    plot_series_w_labels, visualize_num_sample_num_epoch)

RNG = Random()
RNG_SEED = 0xDEADBEEF

TASK = sys.argv[1]
_arch = f"{ARCH}.{TASK}"

EXPERIMENT_NAME = f"3111.{_arch}"
DATASET_SIZES = [125, 250, 500]
MAX_EPOCH = 8


def _train():
    # root/numsample/issue/fold
    path2datasets = {}

    for issue in ISSUES:
        fold2split2samples = load_kfold_primary_frame_samples([issue], KFOLD)
        num_train_sample = len(fold2split2samples[0]["train"])
        for ki in FOLDS_TO_RUN:
            split2samples = fold2split2samples[ki]
            RNG.seed(RNG_SEED)
            RNG.shuffle(split2samples["train"])

            for numsample in DATASET_SIZES:
                if numsample > num_train_sample:
                    continue
                train_samples = split2samples["train"][:numsample]
                valid_samples = split2samples["valid"]
                train_dataset = PrimaryFrameDataset(train_samples)
                valid_dataset = PrimaryFrameDataset(valid_samples)

                path2datasets[
                    join(
                        MODELS_DIR,
                        EXPERIMENT_NAME,
                        f"{numsample:04}_samples",
                        issue,
                        f"fold_{ki}",
                    )
                ] = {
                    "train": train_dataset,
                    "valid": valid_dataset,
                }

    run_experiments(
        _arch,
        path2datasets,
        batchsize=BATCHSIZE,
        max_epochs=MAX_EPOCH,
        save_model=False,
        keep_latest=True,
    )


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
    for epoch in range(MAX_EPOCH):
        reduce_and_save_metrics(
            join(MODELS_DIR, EXPERIMENT_NAME),
            leaf_metric_filename=f"leaf_epoch_{epoch}.json",
            save_filename=f"mean_epoch_{epoch}.json",
        )
    visualize_num_sample_num_epoch(
        join(MODELS_DIR, EXPERIMENT_NAME),
        DATASET_SIZES,
        range(MAX_EPOCH),
        title=f"3111.{_arch}",
        legend_title="num samples",
        xlabel="epoch idx",
        ylabel="valid f1",
    )
