from os.path import exists, join
from typing import List

from config import AUG_SINGLE_SPANS_DIR, FRAMING_DATA_DIR
from tqdm import tqdm

from media_frame_transformer.dataset import TextSample
from media_frame_transformer.utils import load_json


def get_kfold_span_frame_train_samples(
    issues: List[str], k: int, min_span_len: int, augment_sample_weight: float
):
    for issue in issues:
        assert exists(
            join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json")
        ), f"{issue}_{k}_folds.json does not exist, run create_kfold first"

    fold2samples = [[] for _ in range(k)]

    for issue in tqdm(issues):
        frame_span_data = load_json(
            join(AUG_SINGLE_SPANS_DIR, f"{issue}_frame_spans_min{min_span_len}.json")
        )
        kfold_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json"))

        for ki, fold in enumerate(kfold_data["primary_frame"]):
            for id in fold["train"]:
                spans_in_that_article = frame_span_data.get(id, [])
                for span in spans_in_that_article:
                    fold2samples[ki].append(
                        TextSample(
                            text=span["text"],
                            code=span["code"],
                            weight=augment_sample_weight,
                        )
                    )
    return fold2samples


def augment_train_splits(base_fold2split2samples, aug_fold2samples):
    assert len(base_fold2split2samples) == len(aug_fold2samples)
    for ki in range(len(base_fold2split2samples)):
        base_fold2split2samples[ki]["train"].extend(aug_fold2samples[ki])
