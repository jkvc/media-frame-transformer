import sys
from os.path import basename, join, splitext

from media_frame_transformer.dataset.framing.bow_dataset import run_lexicon_experiment
from media_frame_transformer.dataset.framing.definition import ISSUES
from media_frame_transformer.dataset.framing.samples import load_all_framing_samples
from media_frame_transformer.utils import load_json

from config import LEXICON_DIR

_CONFIG_PATH = sys.argv[1]
_CONFIG = load_json(_CONFIG_PATH)

_EXPERIMENT_NAME = "1.all_issues"
_CONFIG_NAME = splitext(basename(_CONFIG_PATH))[0]

_SAVE_DIR = join(LEXICON_DIR, "framing", _EXPERIMENT_NAME, _CONFIG_NAME)

samples = load_all_framing_samples(ISSUES, "train", "primary_frame")
run_lexicon_experiment(
    _CONFIG,
    train_samples=samples,
    valid_samples=samples,
    vocab_size=_CONFIG["vocab_size"],
    logdir=_SAVE_DIR,
)
