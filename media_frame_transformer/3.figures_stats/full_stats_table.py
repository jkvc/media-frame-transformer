# Usage: python <script_name>

from os import makedirs
from os.path import basename, join, realpath

import numpy as np
import pandas as pd
import torch
from config import LEXICON_DIR, MODELS_DIR, OUTPUT_DIR
from media_frame_transformer.datadef.zoo import get_datadef
from media_frame_transformer.utils import load_json

_DATASETS = ["framing", "arxiv"]
_METRIC_FILENAME = "mean_metrics.json"
# _METRIC_FILENAME = "mean_test.json"
_EXP_NAME = "train_single"

_SAVE_ROOT = join(OUTPUT_DIR, "full_stats_table")
makedirs(_SAVE_ROOT, exist_ok=True)

# lexicon stats

_LEXICON_BASE_ARCH = "logreg"
# _LEXICON_ARCHS = ["logreg+kb", "logreg+sn+kb"]
_LEXICON_ARCHS = [
    "logreg+gr",
    "logreg+kb",
    "logreg+kb+gr",
    "logreg+lr",
    "logreg+lr+gr",
    "logreg+sn",
    "logreg+sn+gr",
    "logreg+sn+kb",
    "logreg+sn+kb+gr",
    "logreg+sn+lr",
    "logreg+sn+lr+gr",
]
_ROBERTA_BASE_ARCH = "roberta"
_ROBERTA_ARCHS = [
    "roberta+lr",
    "roberta+kb",
]


def get_valid_accs(metrics):
    return np.array(
        [
            max(
                metrics[source]["mean"].get("f1", 0),
                metrics[source]["mean"].get("valid_f1", 0),
                metrics[source]["mean"].get("valid_f1.best", 0),
            )
            for source in _DATADEF.source_names
        ]
    )


for datasetname in _DATASETS:
    rows = {}
    _DATADEF = get_datadef(datasetname)

    lexicon_metrics = load_json(
        join(LEXICON_DIR, datasetname, _EXP_NAME, _METRIC_FILENAME)
    )
    lexicon_base_accs = np.array(
        [
            lexicon_metrics[_LEXICON_BASE_ARCH][source]["mean"]["valid_f1"]
            for source in _DATADEF.source_names
        ]
    )
    rows[_LEXICON_BASE_ARCH] = {
        "acc": round(lexicon_base_accs.mean(), 3),
        "delta_std": "-",
    }
    for arch in _LEXICON_ARCHS:
        accs = np.array(
            [
                lexicon_metrics[arch][source]["mean"]["valid_f1"]
                for source in _DATADEF.source_names
            ]
        )
        delta = accs - lexicon_base_accs
        rows[arch] = {
            "acc": f"{round(accs.mean(), 3):0.3f}",
            "delta_std": round(delta.std(), 3),
        }

    roberta_base_metrics = load_json(
        join(
            MODELS_DIR,
            datasetname,
            _EXP_NAME,
            _ROBERTA_BASE_ARCH,
            _METRIC_FILENAME,
        )
    )
    roberta_base_accs = get_valid_accs(roberta_base_metrics)
    rows[_ROBERTA_BASE_ARCH] = {
        "acc": round(roberta_base_accs.mean(), 3),
        "delta_std": "-",
    }
    for arch in _ROBERTA_ARCHS:
        metrics = load_json(
            join(MODELS_DIR, datasetname, _EXP_NAME, arch, _METRIC_FILENAME)
        )
        accs = get_valid_accs(metrics)
        delta = accs - roberta_base_accs
        rows[arch] = {
            "acc": f"{round(accs.mean(), 3):0.3f}",
            "delta_std": round(delta.std(), 3),
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.to_csv(join(_SAVE_ROOT, f"{datasetname}.csv"))
