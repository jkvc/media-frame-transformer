from os import makedirs
from os.path import dirname, exists, join
from typing import Dict, List, Optional

import numpy as np
from config import DATA_DIR
from media_frame_transformer.dataset.common import (
    calculate_labelprops,
    get_labelprops_full_split,
)
from media_frame_transformer.dataset.data_sample import DataSample
from media_frame_transformer.utils import load_json, save_json
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

INPUT_N_TOKEN = 512
PAD_TOK_IDX = 1


class RobertaDataset(Dataset):
    def __init__(
        self,
        samples: List[DataSample],
        n_classes: int,
        source_names: List[str],
        source2labelprops: Optional[Dict[str, np.array]] = None,
    ):
        self.samples: List[DataSample] = samples

        if source2labelprops is not None:
            self.source2labelprops = source2labelprops
        else:
            # estimate labelprops in given samples
            self.source2labelprops = calculate_labelprops(
                samples, n_classes, source_names
            )

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

        return {
            "x": x,
            "y": sample.y_idx,
            "labelprops": self.source2labelprops[sample.source_name],
            "source_idx": sample.source_idx,
        }


# def get_kfold_primary_frames_roberta_datasets(
#     issues: List[str],
# ) -> List[Dict[str, List[RobertaDataset]]]:
#     fold2split2samples = load_kfold_framing_samples(issues, task="primary_frame")
#     return fold2split2samples_to_roberta_datasets(fold2split2samples)


# def fold2split2samples_to_roberta_datasets(fold2split2samples):
#     fold2split2datasets = [
#         {
#             split_name: RobertaDataset(split_samples)
#             for split_name, split_samples in split2samples.items()
#         }
#         for split2samples in fold2split2samples
#     ]
#     return fold2split2datasets
