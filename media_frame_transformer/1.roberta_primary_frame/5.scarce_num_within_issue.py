import sys
from os.path import join
from random import Random

import media_frame_transformer.models_roberta  # noqa
from config import (
    BATCHSIZE,
    DATASET_SIZES,
    FOLDS_TO_RUN,
    ISSUES,
    MODELS_DIR,
    RANDOM_SEED,
)
from media_frame_transformer.dataset import PrimaryFrameDataset
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.text_samples import load_kfold_text_samples
from media_frame_transformer.viualization import visualize_num_sample_num_epoch

RNG = Random()

_arch = sys.argv[1]

EXPERIMENT_NAME = f"5.{_arch}"
MAX_EPOCH = 20


def _train():
    # root/numsample/issue/fold
    path2datasets = {}

    for issue in ISSUES:
        fold2split2samples = load_kfold_text_samples([issue], task="primary_frame")
        num_train_sample = len(fold2split2samples[0]["train"])
        for ki in FOLDS_TO_RUN:
            split2samples = fold2split2samples[ki]
            RNG.seed(RANDOM_SEED)
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
