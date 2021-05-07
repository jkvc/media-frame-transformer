from dataclasses import dataclass
from os.path import join
from typing import Dict, List

from config import DATA_DIR
from media_frame_transformer.dataset.arxiv.definition import (
    ARXIV_CATEGORY2IDX,
    year2yidx,
)
from media_frame_transformer.dataset.data_sample import DataSample
from media_frame_transformer.utils import load_json
from tqdm import tqdm


def load_all_arxiv_abstract_samples(
    categories: List[str], split: str
) -> List[DataSample]:
    assert split in ["train", "valid", "test"]

    samples = []
    for c in tqdm(categories):
        ids = load_json(join(DATA_DIR, "arxiv", "splits", f"{c}.{split}.json"))
        raw_data = load_json(join(DATA_DIR, "arxiv", f"{c}.json"))

        for id in ids:
            samples.append(
                DataSample(
                    id=id,
                    text=raw_data[id]["abstract"],
                    y_idx=year2yidx(raw_data[id]["year"]),
                    source_name=c,
                    source_idx=ARXIV_CATEGORY2IDX[c],
                )
            )
    return samples
