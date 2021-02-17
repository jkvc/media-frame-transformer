from dataclasses import dataclass
from os.path import exists, join
from typing import Dict, List, Literal

import numpy as np
import torch
from config import FRAMING_DATA_DIR, ISSUES
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, RobertaTokenizer, RobertaTokenizerFast
from transformers.utils.dummy_pt_objects import TransfoXLLMHeadModel

from media_frame_transformer.utils import load_json

INPUT_N_TOKEN = 512
PAD_TOK_IDX = 1


@dataclass
class TextSample:
    text: str
    code: float
    weight: float = 1


def pad_encoded(x):
    return x + ([PAD_TOK_IDX] * (INPUT_N_TOKEN - len(x)))


def clean_text(text):
    lines = text.split("\n\n")
    lines = lines[3:]  # first 3 lines are id, "PRIMARY", title
    text = "\n".join(lines)
    return text


def load_kfold_primary_frame_samples(
    issues: List[str], k: int
) -> List[Dict[str, List[TextSample]]]:
    for issue in issues:
        assert exists(
            join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json")
        ), f"{issue}_{k}_folds.json does not exist, run create_kfold first"

    fold2split2samples = [{"train": [], "valid": []} for _ in range(k)]

    for issue in tqdm(issues):
        raw_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))
        kfold_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json"))

        for ki, fold in enumerate(kfold_data["primary_frame"]):
            for id in fold["train"]:
                fold2split2samples[ki]["train"].append(
                    TextSample(
                        text=clean_text(raw_data[id]["text"]),
                        code=raw_data[id]["primary_frame"],
                        weight=1,
                    )
                )
            for id in fold["valid"]:
                fold2split2samples[ki]["valid"].append(
                    TextSample(
                        text=clean_text(raw_data[id]["text"]),
                        code=raw_data[id]["primary_frame"],
                        weight=1,
                    )
                )
    return fold2split2samples


def frame_code_to_idx(frame_float: float) -> int:
    # see codes.json, non null frames are [1.?, 15.?], map them to [0, 14]
    assert frame_float != 0
    assert frame_float < 16
    return int(frame_float) - 1


TOKENIZER = RobertaTokenizerFast.from_pretrained("roberta-base")


class PrimaryFrameDataset(Dataset):
    def __init__(self, samples: List[TextSample]):
        self.samples: List[TextSample] = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = np.array(
            TOKENIZER.encode(
                sample.text,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
            )
        )
        y = frame_code_to_idx(sample.code)
        return (x, y, sample.weight)


def get_kfold_primary_frames_datasets(
    issues: List[str], k: int
) -> List[Dict[str, List[PrimaryFrameDataset]]]:
    fold2split2samples = load_kfold_primary_frame_samples(issues, k)
    return fold2split2samples_to_datasets(fold2split2samples)


def fold2split2samples_to_datasets(fold2split2samples):
    fold2split2datasets = [
        {
            split_name: PrimaryFrameDataset(split_samples)
            for split_name, split_samples in split2samples.items()
        }
        for split2samples in fold2split2samples
    ]
    return fold2split2datasets


if __name__ == "__main__":
    fold2split2samples = load_kfold_primary_frame_samples(["climate"], 8)
    for ki, split2samples in enumerate(fold2split2samples):
        train_samples = split2samples["train"]
        train_ds = PrimaryFrameDataset(train_samples)
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=30, num_workers=2)
        for batch in train_loader:
            print(batch)
            break


# def load_kfold(
#     issues: List[str],
#     task: str,
#     k: int,
# ) -> List[Dict[str, Dataset]]:
#     assert task in ["primary_frame"]  # todo support more transforms
#     if task == "primary_frame":
#         label_transform = frame_code_to_idx
#     else:
#         label_transform = lambda x: x  # identity, placeholder

#     for issue in issues:
#         assert exists(
#             join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json")
#         ), f"{issue}_{k}_folds.json does not exist, run create_kfold first"

#     fold2xs = [{"train": [], "valid": []} for _ in range(k)]
#     fold2ys = [{"train": [], "valid": []} for _ in range(k)]

#     for issue in tqdm(issues):
#         raw_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))
#         encoded_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_encoded.json"))
#         kfold_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json"))

#         for ki, fold in enumerate(kfold_data[task]):
#             trainxs = [np.array(pad_encoded(encoded_data[id])) for id in fold["train"]]
#             trainys = [label_transform(raw_data[id][task]) for id in fold["train"]]
#             validxs = [np.array(pad_encoded(encoded_data[id])) for id in fold["valid"]]
#             validys = [label_transform(raw_data[id][task]) for id in fold["valid"]]
#             fold2xs[ki]["train"] += trainxs
#             fold2ys[ki]["train"] += trainys
#             fold2xs[ki]["valid"] += validxs
#             fold2ys[ki]["valid"] += validys

#     results = [
#         {
#             "train": TensorDataset(
#                 torch.LongTensor(fold2xs[ki]["train"]),
#                 torch.LongTensor(fold2ys[ki]["train"]),
#             ),
#             "valid": TensorDataset(
#                 torch.LongTensor(fold2xs[ki]["valid"]),
#                 torch.LongTensor(fold2ys[ki]["valid"]),
#             ),
#         }
#         for ki in range(k)
#     ]
#     return results
