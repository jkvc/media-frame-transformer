from dataclasses import dataclass
from os.path import join
from typing import Dict, List

from config import DATA_DIR, KFOLD
from media_frame_transformer.dataset.framing.definition import (
    ISSUE2IIDX,
    primary_frame_code_to_fidx,
)
from media_frame_transformer.utils import load_json
from tqdm import tqdm

TASKS = ["relevance", "primary_frame", "primary_tone"]


@dataclass
class FramingDataSample:
    id: str
    text: str
    code: float
    frame_idx: int
    issue: str
    issue_idx: int


def load_all_framing_samples(
    issues: List[str], split: str, task: str
) -> List[FramingDataSample]:
    assert split in ["train", "test"]
    assert task in TASKS

    samples = []
    for issue in tqdm(issues):
        ids = load_json(
            join(DATA_DIR, "framing_labeled", f"{issue}_{split}_sets.json")
        )[task]
        raw_data = load_json(join(DATA_DIR, "framing_labeled", f"{issue}_labeled.json"))

        for id in ids:
            samples.append(
                FramingDataSample(
                    id=id,
                    text=clean_text(raw_data[id]["text"]),
                    code=raw_data[id]["primary_frame"],
                    frame_idx=primary_frame_code_to_fidx(raw_data[id]["primary_frame"]),
                    issue=issue,
                    issue_idx=ISSUE2IIDX[issue],
                )
            )
    return samples


def load_kfold_framing_samples(
    issues: List[str], task: str
) -> List[Dict[str, List[FramingDataSample]]]:
    assert task in TASKS

    kidx2split2samples = [{"train": [], "valid": []} for _ in range(KFOLD)]

    samples = load_all_framing_samples(issues, split="train", task=task)
    for issue in tqdm(issues):
        kfold_data = load_json(
            join(DATA_DIR, "framing_labeled", f"{KFOLD}fold", f"{issue}.json")
        )
        for kidx, fold in enumerate(kfold_data[task]):
            for split in ["train", "valid"]:
                ids = set(fold[split])
                selected_samples = [s for s in samples if s.id in ids]
                kidx2split2samples[kidx][split].extend(selected_samples)
    return kidx2split2samples


def clean_text(text):
    lines = text.split("\n\n")
    lines = lines[3:]  # first 3 lines are id, "PRIMARY", title
    text = "\n".join(lines)
    return text
