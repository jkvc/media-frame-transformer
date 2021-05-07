import sys
from os.path import join

from config import ISSUES, LEX_DIR
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.lexicon import run_lexicon_experiment
from media_frame_transformer.text_samples import load_all_text_samples

_arch = sys.argv[1]
_C = 0.025

if __name__ == "__main__":

    for issue in ISSUES:
        print(">>", issue)

        train_samples = load_all_text_samples([issue], "train", "primary_frame")
        valid_issues = [i for i in ISSUES if i != issue]
        valid_samples = load_all_text_samples(valid_issues, "train", "primary_frame")

        run_lexicon_experiment(
            _arch,
            _C,
            train_samples,
            valid_samples,
            join(LEX_DIR, f"6f.{_arch}", f"{issue}"),
        )
    reduce_and_save_metrics(join(LEX_DIR, f"6f.{_arch}"))
