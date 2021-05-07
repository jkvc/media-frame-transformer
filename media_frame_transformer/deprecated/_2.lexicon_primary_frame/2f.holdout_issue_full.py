import sys
from os.path import join

from config import ISSUES, LEX_DIR
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.lexicon import run_lexicon_experiment
from media_frame_transformer.text_samples import load_all_text_samples

_arch = sys.argv[1]

# regularization parameter, found by kfold cross validation
_HOLDOUTISSUE2C = {
    "climate": 0.0143,
    "deathpenalty": 0.0186,
    "guncontrol": 0.0316,
    "immigration": 0.0186,
    "samesex": 0.0230,
    "tobacco": 0.0273,
}

if __name__ == "__main__":

    for holdout_issue in ISSUES:
        print(">>", holdout_issue)

        train_issues = [i for i in ISSUES if i != holdout_issue]
        train_samples = load_all_text_samples(train_issues, "train", "primary_frame")
        valid_samples = load_all_text_samples([holdout_issue], "train", "primary_frame")

        run_lexicon_experiment(
            _arch,
            _HOLDOUTISSUE2C[holdout_issue],
            train_samples,
            valid_samples,
            join(LEX_DIR, f"2f.{_arch}", f"holdout_{holdout_issue}"),
        )
    reduce_and_save_metrics(join(LEX_DIR, f"2f.{_arch}"))
