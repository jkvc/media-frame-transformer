# Usage: python <script_name> <dataset_name> <reg>

import sys
from os.path import basename, join, realpath

import torch
from config import LEXICON_DIR
from media_frame_transformer.datadef.zoo import get_datadef
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.lexicon import eval_lexicon_model, run_lexicon_experiment
from media_frame_transformer.model.logreg_config.grid_search import (
    load_logreg_model_config_all_archs,
)
from media_frame_transformer.utils import read_txt_as_str_list, save_json

_DATASET_NAME = sys.argv[1]
_REG = float(sys.argv[2])

_DATADEF = get_datadef(_DATASET_NAME)

_SCRIPT_PATH = realpath(__file__)
_EXPERIMENT_NAME = basename(_SCRIPT_PATH).replace(".py", "")
_SAVE_ROOT = join(LEXICON_DIR, _DATASET_NAME, _EXPERIMENT_NAME)

_ARCH2CONFIG = load_logreg_model_config_all_archs(
    _DATADEF.n_classes, _DATADEF.n_sources
)
for arch, config in _ARCH2CONFIG.items():
    config["reg"] = _REG  # override with cmd arg

    print("\n")
    print("+" * 30)
    print(arch)
    print("+" * 30)

    savedir = join(_SAVE_ROOT, arch)

    for train_source in _DATADEF.source_names:
        print(">>", train_source)

        holdout_sources = [s for s in _DATADEF.source_names if s != train_source]

        train_samples = _DATADEF.load_splits_func([train_source], ["train"])["train"]
        # valid using holdout issue all samples
        valid_samples = _DATADEF.load_splits_func(holdout_sources, ["train"])["train"]

        run_lexicon_experiment(
            config,
            _DATADEF,
            train_samples=train_samples,
            valid_samples=valid_samples,
            vocab_size=config["vocab_size"],
            logdir=join(savedir, train_source),
            train_labelprop_split="train",
            valid_labelprop_split="train",
        )

        model = torch.load(join(savedir, train_source, "model.pth"))
        vocab = read_txt_as_str_list(join(savedir, train_source, "vocab.txt"))

        test_samples = _DATADEF.load_splits_func(holdout_sources, ["test"])["test"]
        test_metrics = eval_lexicon_model(
            model,
            _DATADEF,
            test_samples,
            vocab,
            use_source_individual_norm=config["use_source_individual_norm"],
            labelprop_split="test",
        )
        save_json(test_metrics, join(savedir, train_source, "leaf_test.json"))

    save_json(config, join(savedir, "config.json"))

reduce_and_save_metrics(_SAVE_ROOT)
reduce_and_save_metrics(_SAVE_ROOT, "leaf_test.json", "mean_test.json")
