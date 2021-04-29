import sys
from os import makedirs
from os.path import join
from random import Random

import numpy as np
import pandas as pd
from config import FOLDS_TO_RUN, ISSUES, LEX_DIR, RANDOM_SEED
from media_frame_transformer.dataset import (
    PRIMARY_FRAME_NAMES,
    get_kfold_primary_frames_datasets,
    get_primary_frame_labelprops_full_split,
)
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.lexicon import run_lexicon_experiment
from media_frame_transformer.text_samples import (
    load_all_text_samples,
    load_kfold_text_samples,
)

RNG = Random()
RNG.seed(RANDOM_SEED)

_arch = sys.argv[1]
_C = float(sys.argv[2])

if __name__ == "__main__":
    kidx2split2samples = load_kfold_text_samples(ISSUES, "primary_frame")
    for kidx, split2samples in enumerate(kidx2split2samples):
        if kidx not in FOLDS_TO_RUN:
            continue
        run_lexicon_experiment(
            _arch,
            _C,
            split2samples["train"],
            split2samples["valid"],
            join(LEX_DIR, f"1.{_arch}.{_C}", str(kidx)),
        )
    reduce_and_save_metrics(join(LEX_DIR, f"1.{_arch}.{_C}"))
