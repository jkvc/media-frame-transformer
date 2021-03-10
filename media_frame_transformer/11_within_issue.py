from os.path import join

from config import ISSUES, MODELS_DIR

from media_frame_transformer.dataset import get_kfold_primary_frames_datasets
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiment_config import (
    ARCH,
    BATCHSIZE,
    FOLDS_TO_RUN,
    KFOLD,
)
from media_frame_transformer.experiments import run_experiments

EXPERIMENT_NAME = f"11.{ARCH}"


def _train():
    path2datasets = {}
    for issue in ISSUES:
        kfold_datasets = get_kfold_primary_frames_datasets([issue], KFOLD)
        for ki in FOLDS_TO_RUN:
            datasets = kfold_datasets[ki]
            path2datasets[
                join(MODELS_DIR, EXPERIMENT_NAME, issue, f"fold_{ki}")
            ] = datasets
    run_experiments(
        ARCH, path2datasets, batchsize=BATCHSIZE, num_early_stop_non_improve_epoch=5
    )


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
