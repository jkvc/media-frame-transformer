from os import makedirs
from os.path import join

from media_frame_transformer.dataset.common import calculate_labelprops
from media_frame_transformer.dataset.framing.definition import (
    ISSUES,
    LABELPROPS_DIR,
    PRIMARY_FRAME_N_CLASSES,
)
from media_frame_transformer.dataset.framing.samples import load_all_framing_samples
from media_frame_transformer.utils import save_json

makedirs(LABELPROPS_DIR, exist_ok=True)

for split in ["train", "test"]:
    samples = load_all_framing_samples(ISSUES, split, "primary_frame")
    source2labelprops = calculate_labelprops(samples, PRIMARY_FRAME_N_CLASSES, ISSUES)
    save_json(
        {issue: labelprops.tolist() for issue, labelprops in source2labelprops.items()},
        join(LABELPROPS_DIR, f"{split}.json"),
    )
