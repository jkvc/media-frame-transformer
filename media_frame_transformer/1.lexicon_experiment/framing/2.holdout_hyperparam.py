# grid search for hyperparem (L1 regularization constant)

import sys
from os.path import basename, join, splitext

from media_frame_transformer.dataset.framing.bow_dataset import run_lexicon_experiment
from media_frame_transformer.dataset.framing.definition import ISSUES
from media_frame_transformer.dataset.framing.samples import load_kfold_framing_samples
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.utils import load_json

from config import LEXICON_DIR

_CONFIG_PATH = sys.argv[1]
_CONFIG = load_json(_CONFIG_PATH)

_EXPERIMENT_NAME = "2.holdout_hyperparam"
_CONFIG_NAME = splitext(basename(_CONFIG_PATH))[0]

_SAVE_DIR = join(LEXICON_DIR, _EXPERIMENT_NAME, _CONFIG_NAME)

_REG_CANDIDATES = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

for reg in _REG_CANDIDATES:
    for unk_issue in ISSUES:
        print(">>", reg, unk_issue)
        savedir = join(_SAVE_DIR, str(reg), unk_issue)

        known_issues = [i for i in ISSUES if i != unk_issue]
        kidx2split2samples = load_kfold_framing_samples(known_issues, "primary_frame")
        # just use the one fold, since we're doing this once for every unk issue
        split2samples = kidx2split2samples[0]

        train_samples = split2samples["train"]
        valid_samples = split2samples["valid"]

        config = {**_CONFIG}
        config["reg"] = reg
        run_lexicon_experiment(
            config,
            train_samples=train_samples,
            valid_samples=valid_samples,
            vocab_size=config["vocab_size"],
            logdir=savedir,
        )
reduce_and_save_metrics(_SAVE_DIR)
