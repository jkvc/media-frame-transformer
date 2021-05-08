from os.path import join
from typing import Dict, List

import numpy as np
from config import DATA_DIR, KFOLD
from media_frame_transformer.datadef.zoo import DatasetDefinition, register_datadef
from media_frame_transformer.dataset.data_sample import DataSample
from media_frame_transformer.utils import load_json
from tqdm import tqdm

CATEGORIES = [
    "Clothing_Shoes_and_Jewelry",
    "Electronics",
    "Home_and_Kitchen",
    "Kindle_Store",
    "Movies_and_TV",
]
CATEGORY2CIDX = {issue: i for i, issue in enumerate(CATEGORIES)}

RATING_NAMES = ["low", "medium", "high"]


def rating_to_ridx(rating: float) -> int:
    # 1:low
    # 2-4:medium
    # 5:high

    assert int(rating) == rating
    assert rating >= 1.0 and rating <= 5.0
    if rating == 1.0:
        return 0
    elif rating in [2.0, 3.0, 4.0]:
        return 1
    elif rating == 5.0:
        return 2
    else:
        raise NotImplementedError()


_LABELPROPS_DIR = join(DATA_DIR, "amazon_subsampled", "labelprops")


def load_labelprops(split):
    return {
        issue: np.array(labelprops)
        for issue, labelprops in load_json(
            join(_LABELPROPS_DIR, f"{split}.json")
        ).items()
    }


def load_all_amazon_review_samples(
    categories: List[str], split: str
) -> List[DataSample]:
    assert split in ["train", "valid", "test"]

    samples = []
    for c in tqdm(categories):
        ids = load_json(
            join(DATA_DIR, "amazon_subsampled", "splits", f"{c}.{split}.json")
        )
        raw_data = load_json(join(DATA_DIR, "amazon_subsampled", f"{c}.json"))

        for id in ids:
            samples.append(
                DataSample(
                    id=id,
                    text=raw_data[id]["reviewText"],
                    # rating=raw_data[id]["overall"],
                    y_idx=rating_to_ridx(raw_data[id]["overall"]),
                    source_name=c,
                    source_idx=CATEGORY2CIDX[c],
                )
            )
    return samples


def load_splits(
    categories: List[str], splits: List[str]
) -> Dict[str, List[DataSample]]:
    return {
        split: load_all_amazon_review_samples(categories, split) for split in splits
    }


register_datadef(
    "amazon",
    DatasetDefinition(
        source_names=CATEGORIES,
        label_names=RATING_NAMES,
        load_splits_func=load_splits,
        load_labelprops_func=load_labelprops,
    ),
)
