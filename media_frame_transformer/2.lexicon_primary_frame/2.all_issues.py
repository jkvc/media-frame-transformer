import sys
from os.path import join

from config import ISSUES, LEX_DIR
from media_frame_transformer.lexicon import run_lexicon_experiment
from media_frame_transformer.text_samples import load_all_text_samples

_arch = sys.argv[1]

if __name__ == "__main__":
    samples = load_all_text_samples(ISSUES, "train", "primary_frame")
    run_lexicon_experiment(_arch, samples, join(LEX_DIR, f"2.{_arch}"), weight_decay=3)
