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

arch = f"{ARCH}_retrieval"
EXPERIMENT_NAME = f"12b.{arch}"


def _train():
    path2datasets = {}
    kfold_datasets = get_kfold_primary_frames_datasets(ISSUES, KFOLD)
    for ki in FOLDS_TO_RUN:
        path2datasets[join(MODELS_DIR, EXPERIMENT_NAME, f"fold_{ki}")] = kfold_datasets[
            ki
        ]
    run_experiments(arch, path2datasets, batchsize=BATCHSIZE)


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
