from numbers import Number
from os.path import exists, join
from typing import List, Literal, Union

from config import AUG_MULTI_SPANS_DIR, AUG_SINGLE_SPANS_DIR, FRAMING_DATA_DIR
from tqdm import tqdm

from media_frame_transformer.dataset import TextSample
from media_frame_transformer.utils import load_json


def get_kfold_single_span_frame_train_samples(
    issues: List[str], k: int, min_span_len: int, augment_sample_weight: float
) -> List[List[TextSample]]:
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


def get_kfold_multi_span_frame_train_samples_predefined_issue(
    issue: Union[str, Literal["all"]],
    k: int,
    aug_set_size_multiplier: Number,
    augment_sample_weight: float,
) -> List[List[TextSample]]:
    frame_span_data = load_json(
        join(AUG_MULTI_SPANS_DIR, f"{issue}_{k}folds_{aug_set_size_multiplier}x.json")
    )

    fold2samples = [[] for _ in range(k)]
    for ki, samples in frame_span_data.items():
        ki = int(ki)
        for sample in samples:
            fold2samples[ki].append(
                TextSample(
                    text=sample["text"],
                    code=sample["code"],
                    weight=augment_sample_weight,
                )
            )
    return fold2samples


def augment_train_splits(
    base_fold2split2samples: List[List[TextSample]],
    aug_fold2samples: List[List[TextSample]],
):
    assert len(base_fold2split2samples) == len(aug_fold2samples)
    for ki in range(len(base_fold2split2samples)):
        base_fold2split2samples[ki]["train"].extend(aug_fold2samples[ki])
