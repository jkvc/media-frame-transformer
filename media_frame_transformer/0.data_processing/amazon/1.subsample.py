import glob
import gzip
import json
from os import makedirs
from os.path import basename, join
from random import randint, shuffle

import numpy as np
from config import DATA_DIR
from media_frame_transformer.utils import ParallelHandler, save_json
from tqdm import tqdm

_N_SAMPLE_PER_CATEGORY = 3000
_KEEP_KEYS = ["overall", "reviewTime", "reviewText"]
_TRAIN_PROP, _VALID_PROP, _TEST_PROP = [0.8, 0.1, 0.1]

_SRC_DATA_DIR = join(DATA_DIR, "amazon_raw")
_DST_DATA_DIR = join(DATA_DIR, "amazon_subsampled")
_SPLITS_DIR = join(_DST_DATA_DIR, "splits")

makedirs(_DST_DATA_DIR, exist_ok=True)
makedirs(_SPLITS_DIR, exist_ok=True)


def process_category(p):
    category_name = basename(p).split("_5")[0]
    # print(category_name)

    with gzip.open(p, "r") as g:
        lines = [l for l in g]

    n_lines = len(lines)
    samples = {}

    while len(samples) < _N_SAMPLE_PER_CATEGORY:
        idx = randint(0, n_lines)
        l = lines[idx]
        s = json.loads(l)
        if not all(k in s for k in _KEEP_KEYS):
            continue
        sample = {k: s[k] for k in _KEEP_KEYS}
        sample_id = f"{s['asin']}.{s['reviewerID']}"
        samples[sample_id] = sample

    save_json(samples, join(_DST_DATA_DIR, f"{category_name}.json"))

    all_sample_ids = list(samples.keys())
    shuffle(all_sample_ids)
    n_train = int(_N_SAMPLE_PER_CATEGORY * _TRAIN_PROP)
    n_valid = int(_N_SAMPLE_PER_CATEGORY * _VALID_PROP)
    n_test = _N_SAMPLE_PER_CATEGORY - n_train - n_valid
    save_json(
        all_sample_ids[:n_train], join(_SPLITS_DIR, f"{category_name}.train.json")
    )
    save_json(
        all_sample_ids[n_train : n_train + n_valid],
        join(_SPLITS_DIR, f"{category_name}.valid.json"),
    )
    save_json(all_sample_ids[-n_test:], join(_SPLITS_DIR, f"{category_name}.test.json"))


raw_data_paths = sorted(glob.glob(join(_SRC_DATA_DIR, "*.json.gz")))
handler = ParallelHandler(process_category)
handler.run([(p,) for p in raw_data_paths])
