import sys
from os.path import basename, join, splitext

from media_frame_transformer.dataset.framing.bow_dataset import run_lexicon_experiment
from media_frame_transformer.dataset.framing.definition import ISSUES
from media_frame_transformer.dataset.framing.samples import load_all_framing_samples
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.utils import load_json

from config import LEXICON_DIR

_CONFIG_PATH = sys.argv[1]
_CONFIG = load_json(_CONFIG_PATH)

_EXPERIMENT_NAME = "3.holdout_issue"
_CONFIG_NAME = splitext(basename(_CONFIG_PATH))[0]

_SAVE_DIR = join(LEXICON_DIR, "framing", _EXPERIMENT_NAME, _CONFIG_NAME)

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
    )

reduce_and_save_metrics(_SAVE_DIR)
