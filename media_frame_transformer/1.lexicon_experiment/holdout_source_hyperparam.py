# Usage: python <script_name> <dataset_name> <logreg_arch>

import sys
from os.path import basename, join, realpath

from config import LEXICON_DIR
from media_frame_transformer.datadef.zoo import get_datadef
from media_frame_transformer.dataset.bow_dataset import run_lexicon_experiment
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.model.logreg_config.grid_search import (
    load_logreg_model_config_all_archs,
)

_DATASET_NAME = sys.argv[1]
_DATADEF = get_datadef(_DATASET_NAME)
_ARCH = sys.argv[2]
_CONFIG = load_logreg_model_config_all_archs(_DATADEF.n_classes, _DATADEF.n_sources)[
    _ARCH
]

_SCRIPT_PATH = realpath(__file__)
_EXPERIMENT_NAME = basename(_SCRIPT_PATH).replace(".py", "")
_SAVE_ROOT = join(LEXICON_DIR, _DATASET_NAME, _EXPERIMENT_NAME, _ARCH)

_L1_REG_CANDIDATES = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

for reg in _L1_REG_CANDIDATES:
    savedir = join(_SAVE_ROOT, str(reg))

    config = {**_CONFIG}
    config["reg"] = reg  # override

    for holdout_source in _DATADEF.source_names:
        print(">>", holdout_source, reg)
        train_sources = [s for s in _DATADEF.source_names if s != holdout_source]
        train_samples = _DATADEF.load_splits_func(train_sources, ["train"])["train"]
        valid_samples = _DATADEF.load_splits_func(train_sources, ["valid"])["valid"]

        run_lexicon_experiment(
            config,
            _DATADEF,
            train_samples=train_samples,
            valid_samples=valid_samples,
            vocab_size=config["vocab_size"],
            logdir=join(savedir, holdout_source),
            train_labelprop_split="train",
            valid_labelprop_split="valid",
        )

reduce_and_save_metrics(_SAVE_ROOT)
