from os.path import join
from typing import Any, Dict, List

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
    # "police",  # FIXME
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


PRIMARY_TONE_NAMES = [
    "Pro",
    "Neutral",
    "Anti",
]


def primary_tone_code_to_yidx(tone_float: float) -> int:
    assert tone_float >= 17 and tone_float < 20
    return int(tone_float) - 17


def code_to_yidx(code: float, task: str) -> int:
    if task == "primary_frame":
        return primary_frame_code_to_fidx(code)
    elif task == "primary_tone":
        return primary_tone_code_to_yidx(code)
    else:
        raise NotImplementedError()


def remove_framing_text_headings(text):
    lines = text.split("\n\n")
    lines = lines[3:]  # first 3 lines are id, "PRIMARY", title
    text = "\n".join(lines)
    return text


def load_all_framing_samples(
    issues: List[str], split: str, task: str
) -> List[DataSample]:
    assert split in ["train", "test"]

    samples = []
    for issue in tqdm(issues):
        ids = load_json(
            join(DATA_DIR, "framing_labeled", f"{issue}_{split}_sets.json")
        )[task]
        raw_data = load_json(join(DATA_DIR, "framing_labeled", f"{issue}_labeled.json"))

        for id in ids:
            samples.append(
                DataSample(
                    id=id,
                    text=remove_framing_text_headings(raw_data[id]["text"]),
                    y_idx=code_to_yidx(raw_data[id][task], task),
                    source_name=issue,
                    source_idx=ISSUE2IIDX[issue],
                )
            )
    return samples


def load_kfold_framing_samples(
    issues: List[str], task: str
) -> List[Dict[str, List[DataSample]]]:
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


def load_splits(
    issues: List[str], splits: List[str], task: str
) -> Dict[str, List[DataSample]]:
    ret = {}

    if "valid" in splits:
        split2samples = load_kfold_framing_samples(issues, task)[0]
        ret["train"] = split2samples["train"]
        ret["valid"] = split2samples["valid"]
    else:
        ret["train"] = load_all_framing_samples(issues, "train", task)

    if "test" in splits:
        ret["test"] = load_all_framing_samples(issues, "test", task)

    ret = {k: v for k, v in ret.items() if k in splits}
    return ret


_LABELPROPS_DIR = join(DATA_DIR, "framing_labeled", "labelprops")


def load_labelprops(split, task):
    if split == "valid":
        split = "train"  # kfold valid and train are the same set
    return {
        issue: np.array(labelprops)
        for issue, labelprops in load_json(
            join(_LABELPROPS_DIR, f"{task}.{split}.json")
        ).items()
    }


register_datadef(
    "framing",
    DatasetDefinition(
        source_names=ISSUES,
        label_names=PRIMARY_FRAME_NAMES,
        load_splits_func=lambda issues, splits: load_splits(
            issues,
            splits,
            "primary_frame",
        ),
        load_labelprops_func=lambda splits: load_labelprops(splits, "primary_frame"),
    ),
)

register_datadef(
    "framing_tone",
    DatasetDefinition(
        source_names=ISSUES,
        label_names=PRIMARY_TONE_NAMES,
        load_splits_func=lambda issues, splits: load_splits(
            issues,
            splits,
            "primary_tone",
        ),
        load_labelprops_func=lambda splits: load_labelprops(splits, "primary_tone"),
    ),
)
