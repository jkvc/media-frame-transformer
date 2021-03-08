from dataclasses import dataclass
from os.path import exists, join
from typing import Dict, List, Literal, Set

import numpy as np
import torch
from config import DATA_DIR, FRAMING_DATA_DIR, ISSUES
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
    issue: str
    subframes: Set[int]
    weight: float = 1


def pad_encoded(x):
    return x + ([PAD_TOK_IDX] * (INPUT_N_TOKEN - len(x)))


def clean_text(text):
    lines = text.split("\n\n")
    lines = lines[3:]  # first 3 lines are id, "PRIMARY", title
    text = "\n".join(lines)
    return text


def load_all_primary_frame_samples(issues: List[str]) -> List[TextSample]:
    samples = []
    for issue in tqdm(issues):
        train_set_ids = load_json(join(FRAMING_DATA_DIR, f"{issue}_train_sets.json"))[
            "primary_frame"
        ]
        raw_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))
        for id in train_set_ids:
            samples.append(
                TextSample(
                    text=clean_text(raw_data[id]["text"]),
                    code=raw_data[id]["primary_frame"],
                    issue=issue,
                    weight=1,
                )
            )
    return samples


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
            for split in ["train", "valid"]:
                for id in fold[split]:
                    item = raw_data[id]
                    subframes = set(
                        frame_code_to_idx(span["code"])
                        for spans in item["annotations"]["framing"].values()
                        for span in spans
                    )
                    fold2split2samples[ki][split].append(
                        TextSample(
                            text=clean_text(item["text"]),
                            code=item["primary_frame"],
                            issue=issue,
                            subframes=subframes,
                            weight=1,
                        )
                    )
    return fold2split2samples


def frame_code_to_idx(frame_float: float) -> int:
    # see codes.json, non null frames are [1.?, 15.?], map them to [0, 14]
    assert frame_float != 0
    assert frame_float < 16
    return int(frame_float) - 1


def label_idx_to_frame_code(idx: int) -> float:
    return float(idx + 1)


def get_issue2labelprop():
    distr = load_json(join(DATA_DIR, "label_distributions.json"))
    issue2labelprop = {
        issue: np.array(props) for issue, props in distr["props"].items()
    }
    return issue2labelprop


def get_issue2idx():
    issue2labelprop = get_issue2labelprop()
    issue2idx = {issue: i for i, issue in enumerate(issue2labelprop.keys())}
    return issue2idx


TOKENIZER = RobertaTokenizerFast.from_pretrained("roberta-base")


class PrimaryFrameDataset(Dataset):
    def __init__(self, samples: List[TextSample]):
        self.samples: List[TextSample] = samples
        self.issue2labelprop = get_issue2labelprop()
        self.issue2idx = get_issue2idx()

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
        subframes = np.zeros((15,))
        for i in sample.subframes:
            subframes[i] = 1
        return {
            "x": x,
            "y": y,
            "weight": sample.weight,
            "subframes": subframes,
            "label_props": self.issue2labelprop[sample.issue],
            "issue_idx": self.issue2idx[sample.issue],
        }


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
    fold2split2samples = load_kfold_primary_frame_samples(["climate", "tobacco"], 8)
    for ki, split2samples in enumerate(fold2split2samples):
        train_samples = split2samples["train"]
        train_ds = PrimaryFrameDataset(train_samples)
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=30, num_workers=2)
        for batch in train_loader:
            print(batch)
            break
