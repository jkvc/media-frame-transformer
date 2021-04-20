# deprecated

from collections import Counter
from os.path import join

import numpy as np
from config import DATA_DIR, ISSUES
from media_frame_transformer.dataset import (
    frame_code_to_idx,
    load_all_primary_frame_samples,
)
from media_frame_transformer.utils import mkdir_overwrite, save_json

SAVE_DIR = join(DATA_DIR, "distributions")

if __name__ == "__main__":
    mkdir_overwrite(SAVE_DIR)

    issue2primary = {}
    issue2secondary = {}
    issue2both = {}

    for issue in ISSUES:
        samples = load_all_primary_frame_samples([issue])

        # primary frame classification
        priframe_counts = Counter([frame_code_to_idx(s.code) for s in samples])
        priframe_counts = np.array([priframe_counts[i] for i in range(15)])
        priframe_props = (priframe_counts / priframe_counts.sum()).tolist()
        issue2primary[issue] = priframe_props

        # secondary frames, agreeed upon at least 2 annotators
        counts = np.zeros((15,))
        for sample in samples:
            counts[list(sample.subframes)] += 1
        prisubframe_likelihood = (counts / len(samples)).tolist()
        issue2secondary[issue] = prisubframe_likelihood

        # both
        counts = np.zeros((15,))
        for sample in samples:
            all_frames = set(sample.subframes).union({frame_code_to_idx(sample.code)})
            all_frames = list(all_frames)
            counts[all_frames] += 1
        prisubframe_likelihood = (counts / len(samples)).tolist()
        issue2both[issue] = prisubframe_likelihood

    save_json(issue2primary, join(SAVE_DIR, "primary.json"))
    save_json(issue2secondary, join(SAVE_DIR, "secondary.json"))
    save_json(issue2both, join(SAVE_DIR, "both.json"))
