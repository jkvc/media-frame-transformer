import shutil
import sys
from os.path import exists, join
from random import Random

from config import ISSUES, MODELS_DIR

from media_frame_transformer.dataset import (
    PrimaryFrameDataset,
    load_kfold_primary_frame_samples,
)
from media_frame_transformer.eval import reduce_and_save_metrics, reduce_tree_inplace
from media_frame_transformer.experiment_config import (
    ARCH,
    BATCHSIZE,
    DATASET_SIZES,
    FOLDS_TO_RUN,
    KFOLD,
)
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.models import (
    freeze_roberta_all_transformer,
    freeze_roberta_module,
)
from media_frame_transformer.utils import load_json, save_json
from media_frame_transformer.viualization import visualize_num_sample_num_epoch

RNG = Random()
RNG_SEED = 0xDEADBEEF


_arch = sys.argv[1]

EXPERIMENT_NAME = f"14.{_arch}"
CHECKPOINT_EXPERIMENT_NAME = f"13f.{_arch}"

MAX_EPOCH = 10


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
                    "checkpoint.pth",
                )

    run_experiments(
        _arch,
        path2datasets,
        path2checkpointpath=path2checkpointpath,
        save_model=False,
        keep_latest=True,
        batchsize=BATCHSIZE,
        max_epochs=MAX_EPOCH,
        skip_train_zeroth_epoch=True,
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
        title=EXPERIMENT_NAME,
        legend_title="num samples",
        xlabel="epoch idx",
        ylabel="valid f1",
    )

    model_root = join(MODELS_DIR, EXPERIMENT_NAME)
    numepoch2metrics = {
        epoch: load_json(join(model_root, f"mean_epoch_{epoch}.json"))
        for epoch in range(MAX_EPOCH)
        if exists(join(model_root, f"mean_epoch_{epoch}.json"))
    }
    numsample2bestvalid = {}
    bestearlystop_metrics = {}

    for numsample in DATASET_SIZES:
        bestearlystop_metrics[numsample] = {}
        # best_valids = []
        for issue in ISSUES:
            bestearlystop_metrics[numsample][issue] = {}
            for ki in FOLDS_TO_RUN:
                valids = []
                for numepoch in range(MAX_EPOCH):
                    if numepoch not in numepoch2metrics:
                        continue
                    valid = numepoch2metrics[numepoch][f"{numsample:04}_samples"][
                        f"holdout_{issue}"
                    ][f"fold_{ki}"]["mean"]["valid_f1"]
                    valids.append(valid)
                best_valid = max(valids)
                bestearlystop_metrics[numsample][issue][ki] = {
                    "mean": {"best_earlystop_valid_f1": best_valid}
                }
                # best_valids.append(best_valid)
        # numsample2bestvalid[numsample] = sum(best_valids) / len(best_valids)

    reduce_tree_inplace(bestearlystop_metrics)
    save_json(bestearlystop_metrics, join(model_root, "best_earlystop.json"))
