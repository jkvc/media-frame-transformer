import sys
from os.path import basename, dirname, join, realpath

from config import LEXICON_DIR
from media_frame_transformer.dataset.amazon.definition import (
    CATEGORIES,
    LABELPROPS_DIR,
    N_CATEGORIES,
    RATING_N_CLASSES,
    RATING_NAMES,
)
from media_frame_transformer.dataset.amazon.samples import (
    load_all_amazon_review_samples,
)
from media_frame_transformer.dataset.bow_dataset import run_lexicon_experiment
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.model.logreg_config.base import load_logreg_model_config

_ARCH = sys.argv[1]
_CONFIG = load_logreg_model_config(_ARCH, RATING_N_CLASSES, N_CATEGORIES)

_SCRIPT_PATH = realpath(__file__)
_EXPERIMENT_NAME = basename(_SCRIPT_PATH).replace(".py", "")
_DATASOURCE_NAME = basename(dirname(_SCRIPT_PATH))
_SAVE_DIR = join(LEXICON_DIR, _DATASOURCE_NAME, _EXPERIMENT_NAME, _ARCH)

for holdout_cat in CATEGORIES:
    print(">>", holdout_cat)
    savedir = join(_SAVE_DIR, holdout_cat)

    train_cats = [c for c in CATEGORIES if c != holdout_cat]
    train_samples = load_all_amazon_review_samples(train_cats, "train")
    valid_samples = load_all_amazon_review_samples([holdout_cat], "train")

    run_lexicon_experiment(
        _CONFIG,
        train_samples=train_samples,
        valid_samples=valid_samples,
        vocab_size=_CONFIG["vocab_size"],
        logdir=savedir,
        source_names=RATING_NAMES,
        labelprop_dir=LABELPROPS_DIR,
        train_labelprop_split="train",
        valid_labelprop_split="train",
    )

reduce_and_save_metrics(_SAVE_DIR)
