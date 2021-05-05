from os import makedirs
from os.path import dirname, exists, join

import numpy as np
from config import DATA_DIR
from media_frame_transformer.dataset.framing.definition import (
    ISSUES,
    PRIMARY_FRAME_N_CLASSES,
    primary_frame_code_to_fidx,
)
from media_frame_transformer.dataset.framing.samples import load_all_framing_samples
from media_frame_transformer.utils import load_json, save_json


def calculate_primary_frame_labelprops(samples):
    issue2labelcounts = {
        issue: (np.zeros((PRIMARY_FRAME_N_CLASSES,)) + 1e-8) for issue in ISSUES
    }
    for s in samples:
        issue2labelcounts[s.issue][primary_frame_code_to_fidx(s.code)] += 1
    return {
        issue: labelcounts / (labelcounts.sum())
        for issue, labelcounts in issue2labelcounts.items()
    }


def get_primary_frame_labelprops_full_split(split):
    labelprops_path = join(
        DATA_DIR, "framing_labeled", "labelprops_primary_frame", f"{split}.json"
    )

    if exists(labelprops_path):
        return {
            issue: np.array(labelprops)
            for issue, labelprops in load_json(labelprops_path).items()
        }
    else:
        samples = load_all_framing_samples(ISSUES, split="train", task="primary_frame")
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
