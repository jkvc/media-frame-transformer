from os import makedirs
from os.path import dirname, exists, join
from typing import Dict, List, Optional

import numpy as np
from config import DATA_DIR, ISSUES, N_CLASSES
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

from media_frame_transformer.text_samples import (
    TextSample,
    load_all_text_samples,
    load_kfold_text_samples,
)
from media_frame_transformer.utils import load_json, save_json

INPUT_N_TOKEN = 512
PAD_TOK_IDX = 1

PRIMARY_FRAME_NAMES = [
    "Economic",
    "Capacity and Resources",
    "Morality",
    "Fairness and Equality",
    "Legality, Constitutionality, Jurisdiction",
    "Policy Prescription and Evaluation",
    "Crime and Punishment",
    "Security and Defense",
    "Health and Safety",
    "Quality of Life",
    "Cultural Identity",
    "Public Sentiment",
    "Political",
    "External Regulation and Reputation",
    "Other",
]


def primary_frame_code_to_cidx(frame_float: float) -> int:
    # see codes.json, non null frames are [1.?, 15.?], map them to [0, 14]
    assert frame_float != 0
    assert frame_float < 16
    return int(frame_float) - 1


def calculate_primary_frame_labelprops(samples):
    issue2labelcounts = {issue: (np.zeros((N_CLASSES,)) + 1e-8) for issue in ISSUES}
    for s in samples:
        issue2labelcounts[s.issue][primary_frame_code_to_cidx(s.code)] += 1
    return {
        issue: labelcounts / (labelcounts.sum())
        for issue, labelcounts in issue2labelcounts.items()
    }


def get_primary_frame_labelprops_full_split(split):
    labelprops_path = join(DATA_DIR, "labelprops_primary_frame", f"{split}.json")

    if exists(labelprops_path):
        return {
            issue: np.array(labelprops)
            for issue, labelprops in load_json(labelprops_path).items()
        }
    else:
        samples = load_all_text_samples(ISSUES, split="train", task="primary_frame")
        issue2labelprops = calculate_primary_frame_labelprops(samples)
        makedirs(dirname(labelprops_path), exist_ok=True)
        save_json(
            {
                issue: labelprops.tolist()
                for issue, labelprops in issue2labelprops.items()
            },
            labelprops_path,
        )
        return issue2labelprops


class PrimaryFrameDataset(Dataset):
    def __init__(
        self,
        samples: List[TextSample],
        labelprops_source: str = "estimated",
    ):
        self.samples: List[TextSample] = samples

        if labelprops_source in {"train"}:
            self.issue2labelprops = get_primary_frame_labelprops_full_split(
                labelprops_source
            )
        elif labelprops_source == "estimated":
            self.issue2labelprops = calculate_primary_frame_labelprops(samples)
        else:
            raise NotImplementedError()

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
        y = primary_frame_code_to_cidx(sample.code)

        return {
            "x": x,
            "y": y,
            "label_distribution": self.issue2labelprops[sample.issue],
        }


def get_kfold_primary_frames_datasets(
    issues: List[str],
) -> List[Dict[str, List[PrimaryFrameDataset]]]:
    fold2split2samples = load_kfold_text_samples(issues, task="primary_frame")
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
