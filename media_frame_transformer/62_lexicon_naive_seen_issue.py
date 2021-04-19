from os import makedirs
from os.path import join

from config import ISSUES, LEX_DIR

from media_frame_transformer.dataset import load_all_primary_frame_samples
from media_frame_transformer.lexicon import build_lexicon

if __name__ == "__main__":
    makedirs(join(LEX_DIR, "62.naive.seen_issue"), exist_ok=True)

    all_samples = load_all_primary_frame_samples(ISSUES)
    df = build_lexicon(all_samples)
    df.to_csv(join(LEX_DIR, "62.naive.seen_issue", "all.csv"))
