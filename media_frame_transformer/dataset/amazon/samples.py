from dataclasses import dataclass
from os.path import join
from typing import Dict, List

from config import DATA_DIR
from media_frame_transformer.dataset.amazon.definition import (
    CATEGORIES,
    CATEGORY2CIDX,
    rating_to_ridx,
)
from media_frame_transformer.dataset.data_sample import DataSample
from media_frame_transformer.utils import load_json
from tqdm import tqdm


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


if __name__ == "__main__":
    samples = load_all_amazon_review_samples(CATEGORIES, "train")
    print(len(samples))
    print(samples[0])
