# Usage: python <script_name> <dataset_name> <n_epoch> <model_arch>

import sys
from os.path import basename, join, realpath

from config import BATCHSIZE, MODELS_DIR
from media_frame_transformer.datadef.zoo import get_datadef
from media_frame_transformer.dataset.roberta_dataset import RobertaDataset
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.model.roberta_config.base import load_roberta_model_config

_DATASET_NAME = sys.argv[1]
_N_TRAIN_EPOCH = int(sys.argv[2])
_ARCH = sys.argv[3]


_DATADEF = get_datadef(_DATASET_NAME)
_CONFIG = load_roberta_model_config(_ARCH, _DATADEF.n_classes, _DATADEF.n_sources)

_SCRIPT_PATH = realpath(__file__)
_EXPERIMENT_NAME = basename(_SCRIPT_PATH).replace(".py", "")
_SAVE_DIR = join(MODELS_DIR, _DATASET_NAME, _EXPERIMENT_NAME, _ARCH)


logdir2datasets = {}

for holdout_source in _DATADEF.source_names:
    print(">>", holdout_source)

    train_sources = [s for s in _DATADEF.source_names if s != holdout_source]
    train_samples = _DATADEF.load_splits_func(train_sources, ["train"])["train"]
    # valid using holdout issue all samples
    valid_samples = _DATADEF.load_splits_func([holdout_source], ["train"])["train"]

    train_dataset = RobertaDataset(
        train_samples,
        n_classes=_DATADEF.n_classes,
        source_names=_DATADEF.source_names,
        source2labelprops=_DATADEF.load_labelprops_func("train"),
    )
    valid_dataset = RobertaDataset(
        valid_samples,
        n_classes=_DATADEF.n_classes,
        source_names=_DATADEF.source_names,
        source2labelprops=_DATADEF.load_labelprops_func("train"),
    )

    logdir2datasets[join(_SAVE_DIR, holdout_source)] = {
        "train": train_dataset,
        "valid": valid_dataset,
    }


run_experiments(
    _CONFIG,
    logdir2datasets,
    batchsize=BATCHSIZE,
    save_model_checkpoint=True,
    # train a fixed number of epoch since we can't use holdout issue as
    # validation data to early stop
    max_epochs=_N_TRAIN_EPOCH,
    num_early_stop_non_improve_epoch=_N_TRAIN_EPOCH,
)

reduce_and_save_metrics(_SAVE_DIR)
for e in range(_N_TRAIN_EPOCH):
    reduce_and_save_metrics(_SAVE_DIR, f"leaf_epoch_{e}.json", f"mean_epoch_{e}.json")