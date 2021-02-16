from os.path import exists, join
from typing import Dict, List, Literal

import numpy as np
import torch
from config import FRAMING_DATA_DIR, ISSUES
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, RobertaTokenizer, RobertaTokenizerFast

from media_frame_transformer.utils import load_json

INPUT_N_TOKEN = 512
PAD_TOK_IDX = 1


def pad_encoded(x):
    return x + ([PAD_TOK_IDX] * (INPUT_N_TOKEN - len(x)))


def get_frame_index(frame_float: float):
    assert frame_float != 0
    return int(frame_float) - 1


def load_kfold(
    issues: List[str],
    task: str,
    k: int,
) -> List[Dict[str, Dataset]]:
    assert task in ["primary_frame"]  # todo support more transforms
    if task == "primary_frame":
        label_transform = get_frame_index
    else:
        label_transform = lambda x: x  # identity, placeholder

    for issue in issues:
        assert exists(
            join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json")
        ), f"{issue}_{k}_folds.json does not exist, run create_kfold first"

    fold2xs = [{"train": [], "valid": []} for _ in range(k)]
    fold2ys = [{"train": [], "valid": []} for _ in range(k)]

    for issue in tqdm(issues):
        raw_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))
        encoded_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_encoded.json"))
        kfold_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json"))

        for ki, fold in enumerate(kfold_data[task]):
            trainxs = [np.array(pad_encoded(encoded_data[id])) for id in fold["train"]]
            trainys = [label_transform(raw_data[id][task]) for id in fold["train"]]
            validxs = [np.array(pad_encoded(encoded_data[id])) for id in fold["valid"]]
            validys = [label_transform(raw_data[id][task]) for id in fold["valid"]]
            fold2xs[ki]["train"] += trainxs
            fold2ys[ki]["train"] += trainys
            fold2xs[ki]["valid"] += validxs
            fold2ys[ki]["valid"] += validys

    results = [
        {
            "train": TensorDataset(
                torch.LongTensor(fold2xs[ki]["train"]),
                torch.LongTensor(fold2ys[ki]["train"]),
            ),
            "valid": TensorDataset(
                torch.LongTensor(fold2xs[ki]["valid"]),
                torch.LongTensor(fold2ys[ki]["valid"]),
            ),
        }
        for ki in range(k)
    ]
    return results


if __name__ == "__main__":
    kfold_datasets = load_kfold(ISSUES, "primary_frame", 8)

    for ki, fold in enumerate(kfold_datasets):
        trainloader = DataLoader(fold["train"], batch_size=1)
        print(len(trainloader))

        # for x, y in trainloader:
        #     print(x.shape, y.shape)
