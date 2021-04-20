# deprecated

from collections import Counter
from os.path import join

import numpy as np
from config import DATA_DIR, FRAMING_DATA_DIR, ISSUES
from media_frame_transformer.dataset import frame_code_to_idx
from media_frame_transformer.utils import load_json, mkdir_overwrite, save_json
from tqdm import tqdm


def get_agreed_subframes(annotator2spans, textlen):
    if len(annotator2spans) < 2:
        return []

    frameidx2indicator = {i: np.zeros((textlen,)) for i in range(15)}
    for annotator, spans in annotator2spans.items():
        for span in spans:
            start, end = span["start"], span["end"]
            indicator = frameidx2indicator[frame_code_to_idx(span["code"])]
            indicator[start:end] = indicator[start:end] + 1
    frame_idxs = [
        i for i, indicator in frameidx2indicator.items() if indicator.max() > 1
    ]
    return frame_idxs


if __name__ == "__main__":
    mkdir_overwrite(join(DATA_DIR, "subframes"))

    issue2frameidx2perc = {}

    for issue in ISSUES:
        raw = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))

        articleid2subframes = {}
        for articleid, article in tqdm(raw.items()):
            annotator2spans = article["annotations"]["framing"]
            textlen = len(article["text"])
            subframes = get_agreed_subframes(annotator2spans, textlen)
            articleid2subframes[articleid] = subframes

        frameidx2perc = [0] * 15
        for frameidx in range(15):
            frameidx2perc[frameidx] = sum(
                1 for subframes in articleid2subframes.values() if frameidx in subframes
            ) / len(articleid2subframes)
        issue2frameidx2perc[issue] = frameidx2perc

        save_json(articleid2subframes, join(DATA_DIR, "subframes", f"{issue}.json"))

    save_json(issue2frameidx2perc, join(DATA_DIR, "subframes", f"percentages.json"))
