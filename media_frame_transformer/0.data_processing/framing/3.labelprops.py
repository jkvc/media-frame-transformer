from os import makedirs
from os.path import join

from media_frame_transformer.datadef.definitions.framing import (
    _LABELPROPS_DIR,
    ISSUES,
    PRIMARY_FRAME_NAMES,
    PRIMARY_TONE_NAMES,
    load_all_framing_samples,
)
from media_frame_transformer.dataset.common import calculate_labelprops
from media_frame_transformer.utils import save_json

makedirs(_LABELPROPS_DIR, exist_ok=True)

# primary frame
for split in ["train", "test"]:
    samples = load_all_framing_samples(ISSUES, split, "primary_frame")
    source2labelprops = calculate_labelprops(samples, len(PRIMARY_FRAME_NAMES), ISSUES)
    save_json(
        {issue: labelprops.tolist() for issue, labelprops in source2labelprops.items()},
        join(_LABELPROPS_DIR, f"primary_frame.{split}.json"),
    )

# primary tone
for split in ["train", "test"]:
    samples = load_all_framing_samples(ISSUES, split, "primary_tone")
    source2labelprops = calculate_labelprops(samples, len(PRIMARY_TONE_NAMES), ISSUES)
    save_json(
        {issue: labelprops.tolist() for issue, labelprops in source2labelprops.items()},
        join(_LABELPROPS_DIR, f"primary_tone.{split}.json"),
    )
