import sys
from os.path import join

import media_frame_transformer.models_roberta
from config import BATCHSIZE, FOLDS_TO_RUN, ISSUES, KFOLD, MODELS_DIR
from media_frame_transformer.dataset import get_kfold_primary_frames_datasets
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiments import run_experiments

_arch = sys.argv[1]
EXPERIMENT_NAME = f"1.{_arch}"


def _train():
    path2datasets = {}
    for issue in ISSUES:
        kfold_datasets = get_kfold_primary_frames_datasets([issue])
        for ki in FOLDS_TO_RUN:
            datasets = kfold_datasets[ki]
            path2datasets[
                join(MODELS_DIR, EXPERIMENT_NAME, issue, f"fold_{ki}")
            ] = datasets
    run_experiments(_arch, path2datasets, batchsize=BATCHSIZE, save_model=False)


if __name__ == "__main__":
    _train()
    reduce_and_save_metrics(join(MODELS_DIR, EXPERIMENT_NAME))
