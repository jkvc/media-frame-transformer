import shutil
import sys
from os.path import join
from random import Random, shuffle

from config import ISSUES, MODELS_DIR

from media_frame_transformer.dataset import (
    PrimaryFrameDataset,
    get_kfold_primary_frames_datasets,
    load_all_primary_frame_samples,
    load_kfold_primary_frame_samples,
)
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiment_config import (
    ARCH,
    BATCHSIZE,
    FOLDS_TO_RUN,
    KFOLD,
)
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.viualization import (
    plot_series_w_labels,
    visualize_num_sample_num_epoch,
)

RNG = Random()
RNG_SEED = 0xDEADBEEF


TASK = sys.argv[1]
_arch = f"{ARCH}.{TASK}"

EXPERIMENT_NAME = f"14.{_arch}"
CHECKPOINT_EXPERIMENT_NAME = f"13.{_arch}"

DATASET_SIZES = [125, 250, 500, 1000]
MAX_EPOCH = 12


def _train():
    # root/numsample/holdout_issue/fold
    path2datasets = {}
    path2checkpointpath = {}

    for holdout_issue in ISSUES:
        model_name = f"holdout_{holdout_issue}"

        fold2split2samples = load_kfold_primary_frame_samples([holdout_issue], KFOLD)
        for ki in FOLDS_TO_RUN:
            split2samples = fold2split2samples[ki]
            RNG.seed(RNG_SEED)
            RNG.shuffle(split2samples["train"])

            for numsample in DATASET_SIZES:
                train_samples = split2samples["train"][:numsample]
                valid_samples = split2samples["valid"]
                train_dataset = PrimaryFrameDataset(train_samples)
                valid_dataset = PrimaryFrameDataset(valid_samples)

                save_dir = join(
                    MODELS_DIR,
                    EXPERIMENT_NAME,
                    f"{numsample:04}_samples",
                    model_name,
                    f"fold_{ki}",
                )
                path2datasets[save_dir] = {
                    "train": train_dataset,
                    "valid": valid_dataset,
                }
                path2checkpointpath[save_dir] = join(
                    MODELS_DIR,
                    CHECKPOINT_EXPERIMENT_NAME,
                    model_name,
                    f"fold_{ki}",
                    "checkpoint.pth",
                )

    run_experiments(
        _arch,
        path2datasets,
        path2checkpointpath=path2checkpointpath,
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

    shutil.copyfile(
        join(MODELS_DIR, CHECKPOINT_EXPERIMENT_NAME, "mean_metrics.json"),
        join(MODELS_DIR, EXPERIMENT_NAME, "mean_metrics_before_adapt.json"),
    )
    visualize_num_sample_num_epoch(
        join(MODELS_DIR, EXPERIMENT_NAME),
        DATASET_SIZES,
        range(MAX_EPOCH),
        title=f"14.{_arch}",
        legend_title="num samples",
        xlabel="epoch idx",
        ylabel="valid f1",
    )
