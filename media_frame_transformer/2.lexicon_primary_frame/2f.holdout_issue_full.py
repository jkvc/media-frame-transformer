import sys
from os.path import join

from config import ISSUES, LEX_DIR
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.lexicon import run_lexicon_experiment
from media_frame_transformer.text_samples import load_all_text_samples

_arch = sys.argv[1]
_C = float(sys.argv[2])

if __name__ == "__main__":

    for holdout_issue in ISSUES:
        train_issues = [i for i in ISSUES if i != holdout_issue]
        train_samples = load_all_text_samples(train_issues, "train", "primary_frame")
        valid_samples = load_all_text_samples([holdout_issue], "train", "primary_frame")

        run_lexicon_experiment(
            _arch,
            _C,
            train_samples,
            valid_samples,
            join(LEX_DIR, f"2f.{_arch}.{_C}", f"holdout_{holdout_issue}"),
        )
    reduce_and_save_metrics(join(LEX_DIR, f"2f.{_arch}.{_C}"))
