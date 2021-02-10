from os.path import exists, join
from typing import Dict, List, Literal

import numpy as np
import torch
from config import FRAMING_DATA_DIR, ISSUES
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, RobertaTokenizer, RobertaTokenizerFast

from media_frame_transformer.utils import load_json


def get_frame_index(frame_float: float):
    assert frame_float != 0
    return int(frame_float) - 1


def get_kfold(issues: List[str], task: str, k: int) -> List[Dict[str, Dataset]]:
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
        token_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_tokenized.json"))
        kfold_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json"))
        for ki, fold in enumerate(kfold_data[task]):
            trainxs = [np.array(token_data[id]) for id in fold["train"]]
            trainys = [label_transform(raw_data[id][task]) for id in fold["train"]]
            validxs = [np.array(token_data[id]) for id in fold["valid"]]
            validys = [label_transform(raw_data[id][task]) for id in fold["valid"]]
            fold2xs[ki]["train"] += trainxs
            fold2ys[ki]["train"] += trainys
            fold2xs[ki]["valid"] += validxs
            fold2ys[ki]["valid"] += validys

    results = [
        {
            "train": TensorDataset(
                torch.FloatTensor(fold2xs[ki]["train"]),
                torch.LongTensor(fold2ys[ki]["train"]),
            ),
            "valid": TensorDataset(
                torch.FloatTensor(fold2xs[ki]["valid"]),
                torch.LongTensor(fold2ys[ki]["valid"]),
            ),
        }
        for ki in range(k)
    ]
    return results


if __name__ == "__main__":
    kfold_datasets = get_kfold(ISSUES, "primary_frame", 8)

    for ki, fold in enumerate(kfold_datasets):
        trainloader = DataLoader(fold["train"], batch_size=1)
        print(len(trainloader))

        # for x, y in trainloader:
        #     print(x.shape, y.shape)

# class PrimaryFrameDataset(Dataset):
#     def __init__(
#         self,
#         issues: List[str],
#         split: Literal["train", "test"],
#         quiet: bool = False,
#     ) -> None:
#         super().__init__()
#         assert split in ["train", "test"]

#         self.all_data = []
#         for issue in tqdm(issues, desc="PrimaryFrameDataset", disable=quiet):
#             assert issue in ISSUES

#             raw_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))
#             token_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_tokenized.json"))
#             testset_ids = set(
#                 load_json(join(FRAMING_DATA_DIR, f"{issue}_test_sets.json"))[
#                     "primary_frame"
#                 ]
#             )

#             for k, v in tqdm(raw_data.items(), disable=True):
#                 if split == "train" and k in testset_ids:
#                     continue
#                 if split == "test" and k not in testset_ids:
#                     continue
#                 if (
#                     v["irrelevant"] == 1
#                     or v["primary_frame"] == None
#                     or v["primary_frame"] == 0
#                 ):
#                     continue
#                 if k not in token_data:
#                     continue

#                 self.all_data.append(
#                     {
#                         "tokens": token_data[k],
#                         "label": PrimaryFrameDataset.get_frame_index(
#                             v["primary_frame"]
#                         ),
#                         "text": v["text"],
#                     }
#                 )

#     def __len__(self) -> int:
#         return len(self.all_data)

#     def __getitem__(self, idx: int):
#         sample = self.all_data[idx]
#         return {
#             "x": np.array(sample["tokens"]),
#             "y": sample["label"],
#         }

#     @staticmethod
#
