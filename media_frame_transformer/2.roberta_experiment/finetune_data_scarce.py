# Usage: python <script_name> ['holdout_adapt'/'single_source_from_scratch'] <dataset_name> <model_arch>

import sys
from os.path import basename, dirname, join, realpath
from random import Random

from config import BATCHSIZE, MODELS_DIR, RANDOM_SEED, ROBERTA_ADAPT_N_SAMPLES
from media_frame_transformer.datadef.zoo import get_datadef
from media_frame_transformer.dataset.roberta_dataset import RobertaDataset
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.model.roberta_config.base import load_roberta_model_config

_N_TRAIN_EPOCH = 10

_EXPERIMENT_NAME = sys.argv[1]
if _EXPERIMENT_NAME == "holdout_adapt":
    load_checkpoint_from_holdout_source = True
elif _EXPERIMENT_NAME == "single_source_from_scratch":
    load_checkpoint_from_holdout_source = False
else:
    raise ValueError()

_DATASET_NAME = sys.argv[2]
_ARCH = sys.argv[3]
_DATADEF = get_datadef(_DATASET_NAME)
_CONFIG = load_roberta_model_config(_ARCH, _DATADEF.n_classes, _DATADEF.n_sources)

_SAVE_DIR = join(MODELS_DIR, _DATASET_NAME, _EXPERIMENT_NAME, _ARCH)
_LOAD_CHECKPOINT_DIR = join(MODELS_DIR, _DATASET_NAME, "holdout_source", _ARCH)

_RNG = Random()


logdir2datasets, logdir2checkpointpath = {}, {}


for adapt_source in _DATADEF.source_names:
    split2samples = _DATADEF.load_splits_func([adapt_source], ["train", "valid"])
    train_samples, valid_samples = split2samples["train"], split2samples["valid"]

    _RNG.seed(RANDOM_SEED)
    _RNG.shuffle(train_samples)

    for nsample in ROBERTA_ADAPT_N_SAMPLES:
        selected_train_samples = train_samples[:nsample]
        train_dataset = RobertaDataset(
            selected_train_samples,
            n_classes=_DATADEF.n_classes,
            source_names=_DATADEF.source_names,
            source2labelprops=_DATADEF.load_labelprops_func("train"),
        )
        valid_dataset = RobertaDataset(
            valid_samples,
            n_classes=_DATADEF.n_classes,
            source_names=_DATADEF.source_names,
            source2labelprops=_DATADEF.load_labelprops_func("valid"),
        )

        logdir = join(_SAVE_DIR, f"{nsample:04}_samples", adapt_source)
        logdir2datasets[logdir] = {
            "train": train_dataset,
            "valid": valid_dataset,
        }
        logdir2checkpointpath[logdir] = join(
            _LOAD_CHECKPOINT_DIR, adapt_source, "checkpoint.pth"
        )

# if set load_checkpoint_from_holdout_source to None, we use fresh model instead of load checkpoint
if load_checkpoint_from_holdout_source:
    print(">> will load checkpoints from", _LOAD_CHECKPOINT_DIR)
else:
    print(">> will use off-the-shelf fresh roberta")

run_experiments(
    _CONFIG,
    logdir2datasets=logdir2datasets,
    logdir2checkpointpath=(
        logdir2checkpointpath if load_checkpoint_from_holdout_source else None
    ),
    save_model_checkpoint=False,
    keep_latest=True,
    batchsize=BATCHSIZE,
    max_epochs=_N_TRAIN_EPOCH,
    num_early_stop_non_improve_epoch=_N_TRAIN_EPOCH,
    skip_train_zeroth_epoch=True,
)

reduce_and_save_metrics(_SAVE_DIR)
for e in range(_N_TRAIN_EPOCH):
    reduce_and_save_metrics(_SAVE_DIR, f"leaf_epoch_{e}.json", f"mean_epoch_{e}.json")
