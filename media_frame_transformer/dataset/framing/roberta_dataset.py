from os import makedirs
from os.path import dirname, exists, join
from typing import Dict, List, Optional

import numpy as np
from config import DATA_DIR
from media_frame_transformer.dataset.framing.common import (
    calculate_primary_frame_labelprops,
    get_primary_frame_labelprops_full_split,
)
from media_frame_transformer.dataset.framing.definition import (
    ISSUES,
    N_ISSUES,
    PRIMARY_FRAME_N_CLASSES,
    primary_frame_code_to_fidx,
)
from media_frame_transformer.dataset.framing.samples import (
    DataSample,
    load_all_framing_samples,
    load_kfold_framing_samples,
)
from media_frame_transformer.utils import load_json, save_json
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

INPUT_N_TOKEN = 512
PAD_TOK_IDX = 1


class PrimaryFrameRobertaDataset(Dataset):
    def __init__(
        self,
        samples: List[DataSample],
        labelprops_source: str = "estimated",
    ):
        self.samples: List[DataSample] = samples

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
        y = primary_frame_code_to_fidx(sample.code)

        return {
            "x": x,
            "y": y,
            "label_distribution": self.issue2labelprops[sample.issue],
        }


def get_kfold_primary_frames_roberta_datasets(
    issues: List[str],
) -> List[Dict[str, List[PrimaryFrameRobertaDataset]]]:
    fold2split2samples = load_kfold_framing_samples(issues, task="primary_frame")
    return fold2split2samples_to_roberta_datasets(fold2split2samples)


def fold2split2samples_to_roberta_datasets(fold2split2samples):
    fold2split2datasets = [
        {
            split_name: PrimaryFrameRobertaDataset(split_samples)
            for split_name, split_samples in split2samples.items()
        }
        for split2samples in fold2split2samples
    ]
    return fold2split2datasets
