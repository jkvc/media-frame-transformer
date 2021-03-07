from collections import Counter
from os.path import join

from config import DATA_DIR, ISSUES

from media_frame_transformer.dataset import (
    frame_code_to_idx,
    load_all_primary_frame_samples,
)
from media_frame_transformer.utils import save_json

if __name__ == "__main__":
    issue2counts = {}
    issue2props = {}

    for issue in ISSUES:
        samples = load_all_primary_frame_samples([issue])
        counts = Counter([frame_code_to_idx(s.code) for s in samples])
        counts = [counts[i] for i in range(15)]
        issue2counts[issue] = counts
        issue2props[issue] = [c / sum(counts) for c in counts]

    save_json(
        {
            "counts": issue2counts,
            "props": issue2props,
        },
        join(DATA_DIR, "label_distributions.json"),
    )
