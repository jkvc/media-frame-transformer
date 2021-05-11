# Usage: python <script_name> <dataset_name>

import sys
from os.path import basename, join, realpath

import pandas as pd
from config import BATCHSIZE, LEXICON_DIR, MODELS_DIR
from media_frame_transformer.datadef.zoo import get_datadef
from media_frame_transformer.dataset.bow_dataset import run_lexicon_experiment
from media_frame_transformer.dataset.roberta_dataset import RobertaDataset
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.model.logreg_config.base import load_logreg_model_config
from media_frame_transformer.model.logreg_config.grid_search import (
    load_logreg_model_config_all_archs,
)
from media_frame_transformer.model.roberta_config.base import load_roberta_model_config

_DATASET_NAME = sys.argv[1]
_DATADEF = get_datadef(_DATASET_NAME)

_SCRIPT_PATH = realpath(__file__)
_EXPERIMENT_NAME = basename(_SCRIPT_PATH).replace(".py", "")
_SAVE_ROOT = join(LEXICON_DIR, _DATASET_NAME, _EXPERIMENT_NAME)

_ARCH2CONFIG = load_logreg_model_config_all_archs(
    _DATADEF.n_classes, _DATADEF.n_sources
)
for arch, config in _ARCH2CONFIG.items():
    print("\n")
    print("+" * 30)
    print(arch)
    print("+" * 30)

    savedir = join(_SAVE_ROOT, arch)
    for holdout_source in _DATADEF.source_names:
        print(">>", holdout_source)
        train_sources = [s for s in _DATADEF.source_names if s != holdout_source]
        train_samples = _DATADEF.load_splits_func(train_sources, ["train"])["train"]
        # valid using holdout issue all samples
        valid_samples = _DATADEF.load_splits_func([holdout_source], ["train"])["train"]

        run_lexicon_experiment(
            config,
            _DATADEF,
            train_samples=train_samples,
            valid_samples=valid_samples,
            vocab_size=config["vocab_size"],
            logdir=join(savedir, holdout_source),
            train_labelprop_split="train",
            valid_labelprop_split="train",
        )

reduce_and_save_metrics(_SAVE_ROOT)
