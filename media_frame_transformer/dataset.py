from os.path import join
from typing import Dict, List, Optional

import numpy as np
from config import DATA_DIR, ISSUES
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

from media_frame_transformer.text_samples import TextSample, load_kfold_text_samples
from media_frame_transformer.utils import load_json

INPUT_N_TOKEN = 512
PAD_TOK_IDX = 1
N_CLASSES = 15


def primary_frame_code_to_cidx(frame_float: float) -> int:
    # see codes.json, non null frames are [1.?, 15.?], map them to [0, 14]
    assert frame_float != 0
    assert frame_float < 16
    return int(frame_float) - 1


CODES = None


def idx_to_frame_name(idx) -> str:
    global CODES
    if CODES == None:
        CODES = load_json(join(DATA_DIR, "framing_labeled", "codes.json"))
    return CODES[f"{idx+1}.0"]


def load_labelprops(source_name):
    return {
        issue: np.array(labelprops)
        for issue, labelprops in load_json(
            join(DATA_DIR, "labelprops", f"{source_name}.json")
        )
    }


class PrimaryFrameDataset(Dataset):
    def __init__(
        self,
        samples: List[TextSample],
        labelprops_source: str = "estimated",
    ):
        self.samples: List[TextSample] = samples

        if labelprops_source in {"train_all"}:
            self.issue2labelprops = load_labelprops(labelprops_source)
        elif labelprops_source == "estimated":
            issue2labelcounts = {
                issue: (np.zeros((N_CLASSES,)) + 1e-8) for issue in ISSUES
            }
            for s in samples:
                issue2labelcounts[s.issue][primary_frame_code_to_cidx(s.code)] += 1
            self.issue2labelprops = {
                issue: labelcounts / (labelcounts.sum())
                for issue, labelcounts in issue2labelcounts.items()
            }
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
