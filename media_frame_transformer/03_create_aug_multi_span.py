from collections import defaultdict
from os import mkdir
from os.path import exists, join
from pprint import pprint

import numpy as np
import pandas as pd
from config import AUG_MULTI_SPANS_DIR, AUG_SINGLE_SPANS_DIR, FRAMING_DATA_DIR, ISSUES
from numpy.random import choice
from tqdm import tqdm, trange

from media_frame_transformer.dataset import frame_code_to_idx, label_idx_to_frame_code
from media_frame_transformer.utils import ParallelHandler, load_json, save_json

KFOLD = 8
AUG_SET_SIZE_MULTIPLIER = 4

MAX_SAMPLE_NUMCHAR = 500
MIN_SAMPLE_NUMCHAR = 30


def sample_single_issue(issue):
    articleid2samples = load_json(
        join(AUG_SINGLE_SPANS_DIR, f"{issue}_frame_spans_min30.json")
    )
    kfolds = load_json(join(FRAMING_DATA_DIR, f"{issue}_8_folds.json"))["primary_frame"]

    ki2augsamples = {}
    for ki, fold in enumerate(kfolds):
        print(">>", issue, ki)
        train_article_ids = fold["train"]
        label2samples = defaultdict(list)
        for article_id in train_article_ids:
            if article_id not in articleid2samples:
                # that article has no labeled spans
                continue
            samples = articleid2samples[article_id]
            for sample in samples:
                label2samples[frame_code_to_idx(sample["code"])].append(sample)

        # for label, samples in sorted(label2samples.items()):
        #     print(label, len(samples))
        labels = sorted(list(label2samples.keys()))
        weights = np.array([len(label2samples[label]) for label in labels])
        weights = weights / weights.sum()

        num_orig_samples = len(train_article_ids)
        num_aug_samples = int(num_orig_samples * AUG_SET_SIZE_MULTIPLIER)
        aug_samples = []
        for _ in range(num_aug_samples):
            label_choice = choice(labels, p=weights)
            span_candidates = label2samples[label_choice]

            chosen_spans = []
            chosen_spans_len = 0
            while True:
                chosen_span = choice(span_candidates)["text"]
                chosen_span_len = len(chosen_span)
                if (
                    chosen_span_len + chosen_spans_len > MAX_SAMPLE_NUMCHAR
                    and chosen_span_len > MIN_SAMPLE_NUMCHAR
                ):
                    break
                chosen_spans.append(chosen_span)
                chosen_spans_len += chosen_span_len
            aug_samples.append(
                {
                    "text": " ".join(chosen_spans),
                    "code": label_idx_to_frame_code(label_choice),
                }
            )

        ki2augsamples[ki] = aug_samples
        print("--", issue, ki)

    save_json(
        ki2augsamples,
        join(
            AUG_MULTI_SPANS_DIR,
            f"{issue}_{KFOLD}folds_{AUG_SET_SIZE_MULTIPLIER}x.json",
        ),
    )


if __name__ == "__main__":
    if not exists(AUG_MULTI_SPANS_DIR):
        mkdir(AUG_MULTI_SPANS_DIR)

    handler = ParallelHandler(sample_single_issue)
    handler.run(ISSUES)
