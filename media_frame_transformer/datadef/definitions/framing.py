from os.path import join
from typing import Dict, List

import numpy as np
from config import DATA_DIR, KFOLD
from media_frame_transformer.datadef.zoo import DatasetDefinition, register_datadef
from media_frame_transformer.dataset.data_sample import DataSample
from media_frame_transformer.utils import load_json
from tqdm import tqdm

ISSUES = [
    "climate",
    "deathpenalty",
    "guncontrol",
    "immigration",
    "samesex",
    "tobacco",
]
ISSUE2IIDX = {issue: i for i, issue in enumerate(ISSUES)}

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


def primary_frame_code_to_fidx(frame_float: float) -> int:
    # see codes.json, non null frames are [1.?, 15.?], map them to [0, 14]
    assert frame_float != 0
    assert frame_float < 16
    return int(frame_float) - 1


def remove_framing_text_headings(text):
    lines = text.split("\n\n")
    lines = lines[3:]  # first 3 lines are id, "PRIMARY", title
    text = "\n".join(lines)
    return text


def load_all_framing_samples(issues: List[str], split: str) -> List[DataSample]:
    assert split in ["train", "test"]

    samples = []
    for issue in tqdm(issues):
        ids = load_json(
            join(DATA_DIR, "framing_labeled", f"{issue}_{split}_sets.json")
        )["primary_frame"]
        raw_data = load_json(join(DATA_DIR, "framing_labeled", f"{issue}_labeled.json"))

        for id in ids:
            samples.append(
                DataSample(
                    id=id,
                    text=remove_framing_text_headings(raw_data[id]["text"]),
                    # code=raw_data[id]["primary_frame"],
                    y_idx=primary_frame_code_to_fidx(raw_data[id]["primary_frame"]),
                    source_name=issue,
                    source_idx=ISSUE2IIDX[issue],
                )
            )
    return samples


def load_kfold_framing_samples(issues: List[str]) -> List[Dict[str, List[DataSample]]]:
    kidx2split2samples = [{"train": [], "valid": []} for _ in range(KFOLD)]

    samples = load_all_framing_samples(issues, split="train")
    for issue in tqdm(issues):
        kfold_data = load_json(
            join(DATA_DIR, "framing_labeled", f"{KFOLD}fold", f"{issue}.json")
        )
        for kidx, fold in enumerate(kfold_data["primary_frame"]):
            for split in ["train", "valid"]:
                ids = set(fold[split])
                selected_samples = [s for s in samples if s.id in ids]
                kidx2split2samples[kidx][split].extend(selected_samples)
    return kidx2split2samples


def load_splits(issues: List[str], splits: List[str]):
    ret = {}

    if "valid" in splits:
        split2samples = load_kfold_framing_samples(issues)[0]
        ret["train"] = split2samples["train"]
        ret["valid"] = split2samples["valid"]
    else:
        ret["train"] = load_all_framing_samples(issues, "train")

    if "test" in splits:
        ret["test"] = load_all_framing_samples(issues, "test")

    ret = {k: v for k, v in ret.items() if k in splits}
    return ret


_LABELPROPS_DIR = join(DATA_DIR, "framing_labeled", "labelprops")


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
    "framing",
    DatasetDefinition(
        source_names=ISSUES,
        label_names=PRIMARY_FRAME_NAMES,
        load_splits_func=load_splits,
        load_labelprops_func=load_labelprops,
    ),
)
