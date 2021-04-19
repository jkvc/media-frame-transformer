from os import makedirs
from os.path import join

from config import ISSUES, LEX_DIR

from media_frame_transformer.dataset import load_all_primary_frame_samples
from media_frame_transformer.lexicon import build_lexicon

if __name__ == "__main__":
    makedirs(join(LEX_DIR, "61.naive.within_issue"), exist_ok=True)

    for issue in ISSUES:
        all_samples = load_all_primary_frame_samples([issue])
        df = build_lexicon(all_samples)
        df.to_csv(join(LEX_DIR, "61.naive.within_issue", f"{issue}.csv"))
