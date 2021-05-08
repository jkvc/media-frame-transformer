from os.path import join
from typing import Dict, List

import numpy as np
from config import DATA_DIR, KFOLD
from media_frame_transformer.datadef.zoo import DatasetDefinition, register_datadef
from media_frame_transformer.dataset.data_sample import DataSample
from media_frame_transformer.utils import load_json
from tqdm import tqdm

ARXIV_CATEGORIES = [
    "cs.AI",
    "cs.CL",  # computation and language
    "cs.CV",
    "cs.LG",  # machine learning
    "cs.NE",  # neural
    "cs.SI",  # social and information network
]
ARXIV_CATEGORY2IDX = {c: i for i, c in enumerate(ARXIV_CATEGORIES)}

YEARRANGE2BOUNDS = {
    "upto2008": (0, 2008),
    "2009-2014": (2009, 2014),
    "2015-2018": (2015, 2018),
    "2019after": (2019, 6969),
}
YEARRANGE_NAMES = list(YEARRANGE2BOUNDS.keys())


def year2yidx(year: int) -> int:
    for i, yearrange_name in enumerate(YEARRANGE_NAMES):
        lb, ub = YEARRANGE2BOUNDS[yearrange_name]
        if year >= lb and year <= ub:
            return i
    raise ValueError()


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


def load_splits(
    categories: List[str], splits: List[str]
) -> Dict[str, List[DataSample]]:
    return {
        split: load_all_arxiv_abstract_samples(categories, split) for split in splits
    }


_LABELPROPS_DIR = join(DATA_DIR, "arxiv", "labelprops")


def load_labelprops(split):
    if split == "valid":
        split = "train"  # kfold valid and train are the same set
    return {
        issue: np.array(labelprops)
        for issue, labelprops in load_json(
            join(_LABELPROPS_DIR, f"{split}.json")
        ).items()
    }


register_datadef(
    "arxiv",
    DatasetDefinition(
        source_names=ARXIV_CATEGORIES,
        label_names=YEARRANGE_NAMES,
        load_splits_func=load_splits,
        load_labelprops_func=load_labelprops,
    ),
)
