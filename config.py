from os.path import dirname, join, realpath

REPO_ROOT = dirname(realpath(__file__))
DATA_DIR = join(REPO_ROOT, "data")
# MODELS_DIR = join(REPO_ROOT, "models")
MODELS_DIR = "/juice/scr/jkc1/models"  # cluster

FRAMING_DATA_DIR = join(DATA_DIR, "framing_labeled")
AUG_SINGLE_SPANS_DIR = join(DATA_DIR, "aug_single_spans")
AUG_MULTI_SPANS_DIR = join(DATA_DIR, "aug_multi_spans")


ISSUES = [
    "climate",
    "deathpenalty",
    "guncontrol",
    "immigration",
    "samesex",
    "tobacco",
]
