# grid search for hyperparem (L1 regularization constant)
import sys
from os.path import basename, join, splitext

from media_frame_transformer.dataset.amazon.definition import (
    CATEGORIES,
    LABELPROPS_DIR,
    RATING_NAMES,
)
from media_frame_transformer.dataset.amazon.samples import (
    load_all_amazon_review_samples,
)
from media_frame_transformer.dataset.bow_dataset import run_lexicon_experiment
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.utils import load_json

from config import LEXICON_DIR

_CONFIG_PATH = sys.argv[1]
_CONFIG = load_json(_CONFIG_PATH)

_EXPERIMENT_NAME = "2.hyperparam"
_CONFIG_NAME = splitext(basename(_CONFIG_PATH))[0]

_SAVE_DIR = join(LEXICON_DIR, "amazon", _EXPERIMENT_NAME, _CONFIG_NAME)

_REG_CANDIDATES = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

for reg in _REG_CANDIDATES:
    for holdout_cat in CATEGORIES:
        print(">>", reg, holdout_cat)

        savedir = join(_SAVE_DIR, str(reg), holdout_cat)

        train_cats = [c for c in CATEGORIES if c != holdout_cat]
        train_samples = load_all_amazon_review_samples(train_cats, "train")
        valid_samples = load_all_amazon_review_samples(train_cats, "valid")

        config = {**_CONFIG}
        config["reg"] = reg

        run_lexicon_experiment(
            config,
            train_samples=train_samples,
            valid_samples=valid_samples,
            vocab_size=config["vocab_size"],
            logdir=savedir,
            source_names=RATING_NAMES,
            labelprop_dir=LABELPROPS_DIR,
        )
reduce_and_save_metrics(_SAVE_DIR)
