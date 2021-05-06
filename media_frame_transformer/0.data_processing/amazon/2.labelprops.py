from os import makedirs
from os.path import join

from media_frame_transformer.dataset.amazon.definition import (
    CATEGORIES,
    LABELPROPS_DIR,
    RATING_N_CLASSES,
)
from media_frame_transformer.dataset.amazon.samples import (
    load_all_amazon_review_samples,
)
from media_frame_transformer.dataset.common import calculate_labelprops
from media_frame_transformer.utils import save_json

makedirs(LABELPROPS_DIR, exist_ok=True)

for split in ["train", "valid", "test"]:
    samples = load_all_amazon_review_samples(CATEGORIES, split)
    source2labelprops = calculate_labelprops(samples, RATING_N_CLASSES, CATEGORIES)
    save_json(
        {issue: labelprops.tolist() for issue, labelprops in source2labelprops.items()},
        join(LABELPROPS_DIR, f"{split}.json"),
    )
