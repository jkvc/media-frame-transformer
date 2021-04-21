from collections import Counter
from os import makedirs
from os.path import join

from config import DATA_DIR, ISSUES
from media_frame_transformer.dataset import PrimaryFrameDataset
from media_frame_transformer.text_samples import load_all_text_samples
from media_frame_transformer.utils import save_json

SAVE_DIR = join(DATA_DIR, "labelprops")

if __name__ == "__main__":
    makedirs(SAVE_DIR, exist_ok=True)

    samples = load_all_text_samples(ISSUES, split="train", task="primary_frame")
    dataset = PrimaryFrameDataset(
        samples,
        labelprops_source="estimated",  # estimate from all train samples
    )

    save_json(
        {
            issue: labelprops.tolist()
            for issue, labelprops in dataset.issue2labelprops.items()
        },
        join(SAVE_DIR, "train_all.json"),
    )
