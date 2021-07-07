# Usage: python <script_name> <dataset_name> <n_epoch> <model_arch>

import sys
from os.path import basename, exists, join, realpath

import torch
from config import BATCHSIZE, MODELS_DIR
from media_frame_transformer.datadef.zoo import get_datadef
from media_frame_transformer.dataset.roberta_dataset import RobertaDataset
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.learning import do_valid
from media_frame_transformer.model.roberta_config.base import load_roberta_model_config
from media_frame_transformer.utils import DEVICE, save_json

_DATASET_NAME = sys.argv[1]
_N_TRAIN_EPOCH = int(sys.argv[2])
_ARCH = sys.argv[3]


_DATADEF = get_datadef(_DATASET_NAME)
_CONFIG = load_roberta_model_config(_ARCH, _DATADEF.n_classes, _DATADEF.n_sources)

_SCRIPT_PATH = realpath(__file__)
_EXPERIMENT_NAME = basename(_SCRIPT_PATH).replace(".py", "")
_SAVE_DIR = join(MODELS_DIR, _DATASET_NAME, _EXPERIMENT_NAME, _ARCH)

# setup and run training

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


# setup and run test set

for holdout_source in _DATADEF.source_names:
    save_metric_path = join(_SAVE_DIR, holdout_source, "leaf_test.json")
    if exists(save_metric_path):
        print(">> skip test", holdout_source)
        continue
    else:
        print(">> test", holdout_source)

    test_samples = _DATADEF.load_splits_func([holdout_source], ["test"])["test"]
    test_dataset = RobertaDataset(
        test_samples,
        n_classes=_DATADEF.n_classes,
        source_names=_DATADEF.source_names,
        source2labelprops=_DATADEF.load_labelprops_func("test"),
    )

    checkpointpath = join(_SAVE_DIR, holdout_source, "checkpoint.pth")
    model = torch.load(checkpointpath).to(DEVICE)
    test_metrics = do_valid(model, test_dataset)

    save_json(test_metrics, save_metric_path)

reduce_and_save_metrics(_SAVE_DIR, f"leaf_test.json", f"mean_test.json")
