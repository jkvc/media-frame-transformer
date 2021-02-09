from os.path import dirname, join, realpath

REPO_ROOT = dirname(realpath(__file__))
DATA_DIR = join(REPO_ROOT, "data")

FRAMING_DATA_DIR = join(DATA_DIR, "framing_labeled")

ISSUES = [
    "climate",
    "deathpenalty",
    "guncontrol",
    "immigration",
    "samesex",
    "tobacco",
]
