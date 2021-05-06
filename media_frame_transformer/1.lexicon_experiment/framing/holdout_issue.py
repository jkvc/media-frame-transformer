import sys
from os.path import basename, dirname, join, realpath

from config import LEXICON_DIR
from media_frame_transformer.dataset.bow_dataset import run_lexicon_experiment
from media_frame_transformer.dataset.framing.definition import (
    ISSUES,
    LABELPROPS_DIR,
    N_ISSUES,
    PRIMARY_FRAME_N_CLASSES,
    PRIMARY_FRAME_NAMES,
)
from media_frame_transformer.dataset.framing.samples import load_all_framing_samples
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.model.logreg_config.base import load_logreg_model_config

_ARCH = sys.argv[1]
_CONFIG = load_logreg_model_config(_ARCH, PRIMARY_FRAME_N_CLASSES, N_ISSUES)

_SCRIPT_PATH = realpath(__file__)
_EXPERIMENT_NAME = basename(_SCRIPT_PATH).replace(".py", "")
_DATASOURCE_NAME = basename(dirname(_SCRIPT_PATH))
_SAVE_DIR = join(LEXICON_DIR, _DATASOURCE_NAME, _EXPERIMENT_NAME, _ARCH)

for holdout_issue in ISSUES:
    print(">>", holdout_issue)
    savedir = join(_SAVE_DIR, holdout_issue)

    train_issues = [i for i in ISSUES if i != holdout_issue]
    train_samples = load_all_framing_samples(train_issues, "train", "primary_frame")
    valid_samples = load_all_framing_samples([holdout_issue], "train", "primary_frame")

    run_lexicon_experiment(
        _CONFIG,
        train_samples=train_samples,
        valid_samples=valid_samples,
        vocab_size=_CONFIG["vocab_size"],
        logdir=savedir,
        source_names=PRIMARY_FRAME_NAMES,
        labelprop_dir=LABELPROPS_DIR,
        train_labelprop_split="train",
        valid_labelprop_split="train",
    )

reduce_and_save_metrics(_SAVE_DIR)
