import sys
from os.path import basename, dirname, join, realpath

from config import BATCHSIZE, MODELS_DIR
from media_frame_transformer.dataset.framing.definition import (
    ISSUES,
    LABELPROPS_DIR,
    N_ISSUES,
    PRIMARY_FRAME_N_CLASSES,
    PRIMARY_FRAME_NAMES,
)
from media_frame_transformer.dataset.framing.samples import load_all_framing_samples
from media_frame_transformer.dataset.roberta_dataset import RobertaDataset
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.model.roberta_config.base import load_roberta_model_config

_N_TRAIN_EPOCH = 6

_ARCH = sys.argv[1]
_CONFIG = load_roberta_model_config(_ARCH, PRIMARY_FRAME_N_CLASSES, N_ISSUES)

_SCRIPT_PATH = realpath(__file__)
_EXPERIMENT_NAME = basename(_SCRIPT_PATH).replace(".py", "")
_DATASOURCE_NAME = basename(dirname(_SCRIPT_PATH))
_SAVE_DIR = join(MODELS_DIR, _DATASOURCE_NAME, _EXPERIMENT_NAME, _ARCH)


path2datasets = {}

for holdout_issue in ISSUES:
    print(">>", holdout_issue)
    savedir = join(_SAVE_DIR, holdout_issue)

    train_issues = [i for i in ISSUES if i != holdout_issue]
    train_samples = load_all_framing_samples(train_issues, "train", "primary_frame")
    valid_samples = load_all_framing_samples([holdout_issue], "train", "primary_frame")

    train_dataset = RobertaDataset(
        train_samples,
        n_classes=PRIMARY_FRAME_N_CLASSES,
        source_names=ISSUES,
        labelprop_split="train",
        labelprop_dir=LABELPROPS_DIR,
    )
    valid_dataset = RobertaDataset(
        valid_samples,
        n_classes=PRIMARY_FRAME_N_CLASSES,
        source_names=ISSUES,
        labelprop_split="train",
        labelprop_dir=LABELPROPS_DIR,
    )

    path2datasets[join(_SAVE_DIR, holdout_issue)] = {
        "train": train_dataset,
        "valid": valid_dataset,
    }

run_experiments(
    _CONFIG,
    path2datasets,
    batchsize=BATCHSIZE,
    save_model_checkpoint=True,
    # train a fixed number of epoch since we can't use holdout issue as
    # validation data to early stop
    max_epochs=_N_TRAIN_EPOCH,
    num_early_stop_non_improve_epoch=_N_TRAIN_EPOCH,
)
reduce_and_save_metrics(_SAVE_DIR)
