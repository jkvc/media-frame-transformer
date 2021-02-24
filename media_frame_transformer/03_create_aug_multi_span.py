import random
from collections import defaultdict
from os import mkdir
from os.path import exists, join
from pprint import pprint

import numpy as np
import pandas as pd
from config import AUG_MULTI_SPANS_DIR, AUG_SINGLE_SPANS_DIR, FRAMING_DATA_DIR, ISSUES
from tqdm import tqdm, trange

from media_frame_transformer.dataset import frame_code_to_idx, label_idx_to_frame_code
from media_frame_transformer.utils import ParallelHandler, load_json, save_json

KFOLD = 8
AUG_SET_SIZE_MULTIPLIER = 1

MAX_SAMPLE_NUMCHAR = 500
MIN_SAMPLE_NUMCHAR = 30

np.random.seed(0xDEADBEEF)
random.seed(0xDEADBEEF)


def sample_single_issue(issues, save_path):

    articleid2samples = {}
    for issue in issues:
        articleid2samples.update(
            load_json(join(AUG_SINGLE_SPANS_DIR, f"{issue}_frame_spans_min30.json"))
        )
    ki2trainids = [[] for _ in range(KFOLD)]
    for issue in issues:
        for ki, fold in enumerate(
            load_json(join(FRAMING_DATA_DIR, f"{issue}_8_folds.json"))["primary_frame"]
        ):
            ki2trainids[ki].extend(fold["train"])

    ki2augsamples = {}
    for ki, train_article_ids in enumerate(ki2trainids):
        print(">>", issues, ki)
        label2samples = defaultdict(list)
        for article_id in train_article_ids:
            if article_id not in articleid2samples:
                # that article has no labeled spans
                continue
            samples = articleid2samples[article_id]
            for sample in samples:
                if (
                    len(sample["text"]) > MIN_SAMPLE_NUMCHAR
                    and len(sample["text"]) < MAX_SAMPLE_NUMCHAR
                ):
                    label2samples[frame_code_to_idx(sample["code"])].append(sample)

        labels = sorted(list(label2samples.keys()))
        weights = np.array([len(label2samples[label]) for label in labels])
        weights = weights / weights.sum()

        num_orig_samples = len(train_article_ids)
        num_aug_samples = int(num_orig_samples * AUG_SET_SIZE_MULTIPLIER)
        aug_samples = []

        for _ in range(num_aug_samples):
            label_choice = np.random.choice(labels, p=weights)
            span_candidates = label2samples[label_choice]

            aug_sample_text = ""
            while True:
                text = random.choice(span_candidates)["text"]
                if len(text) + len(aug_sample_text) > MAX_SAMPLE_NUMCHAR - 1:
                    break
                aug_sample_text += " " + text

            aug_samples.append(
                {
                    "text": aug_sample_text,
                    "code": label_idx_to_frame_code(label_choice),
                }
            )

        ki2augsamples[ki] = aug_samples
        print("--", issues, ki)

    save_json(ki2augsamples, save_path)


if __name__ == "__main__":
    if not exists(AUG_MULTI_SPANS_DIR):
        mkdir(AUG_MULTI_SPANS_DIR)

    handler = ParallelHandler(sample_single_issue)
    params = [
        (
            [issue],
            join(
                AUG_MULTI_SPANS_DIR,
                f"{issue}_{KFOLD}folds_{AUG_SET_SIZE_MULTIPLIER}x.json",
            ),
        )
        for issue in ISSUES
    ]
    # params = []  # fixme
    params.append(
        (
            ISSUES,
            join(
                AUG_MULTI_SPANS_DIR,
                f"all_{KFOLD}folds_{AUG_SET_SIZE_MULTIPLIER}x.json",
            ),
        )
    )
    handler.run(params)
