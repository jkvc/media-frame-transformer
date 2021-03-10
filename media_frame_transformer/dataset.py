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
        articleid2subframes = load_json(join(DATA_DIR, "subframes", f"{issue}.json"))

        for id in train_set_ids:
            item = raw_data[id]
            samples.append(
                TextSample(
                    text=clean_text(raw_data[id]["text"]),
                    code=raw_data[id]["primary_frame"],
                    issue=issue,
                    subframes=set(articleid2subframes[id]),
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
        articleid2subframes = load_json(join(DATA_DIR, "subframes", f"{issue}.json"))

        for ki, fold in enumerate(kfold_data["primary_frame"]):
            for split in ["train", "valid"]:
                for id in fold[split]:
                    item = raw_data[id]
                    fold2split2samples[ki][split].append(
                        TextSample(
                            text=clean_text(item["text"]),
                            code=item["primary_frame"],
                            issue=issue,
                            subframes=set(articleid2subframes[id]),
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


class PrimaryFrameDataset(Dataset):
    def __init__(self, samples: List[TextSample]):
        self.samples: List[TextSample] = samples
        self.issue2labelprop = get_issue2labelprop()
        self.issue2idx = get_issue2idx()
        self.tokenizer = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not self.tokenizer:
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        sample = self.samples[idx]
        x = np.array(
            self.tokenizer.encode(
                sample.text,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
            )
        )

        primary_frame_idx = frame_code_to_idx(sample.code)
        primary_frame_vec = _get_vector_from_idxs([primary_frame_idx])
        subframes_vec = _get_vector_from_idxs(list(sample.subframes))
        retrieval_vec = _get_vector_from_idxs(
            list(sample.subframes.union({primary_frame_idx}))
        )

        return {
            "x": x,
            "weight": sample.weight,
            "primary_frame_idx": primary_frame_idx,
            "primary_frame_vec": primary_frame_vec,
            "subframes": subframes_vec,
            "retrieval": retrieval_vec,
            "label_props": self.issue2labelprop[sample.issue],
            "issue_idx": self.issue2idx[sample.issue],
        }


def _get_vector_from_idxs(idxs):
    vec = np.zeros((15,))
    vec[idxs] = 1
    return vec


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
