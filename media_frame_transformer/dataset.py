from os.path import join
from typing import List, Literal

import numpy as np
import torch
from config import FRAMING_DATA_DIR, ISSUES
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, RobertaTokenizer, RobertaTokenizerFast

from media_frame_transformer.utils import load_json


class PrimaryFrameDataset(Dataset):
    def __init__(
        self,
        issues: List[str],
        split: Literal["train", "test"],
        quiet: bool = False,
    ) -> None:
        super().__init__()
        assert split in ["train", "test"]

        self.all_data = []
        for issue in tqdm(issues, desc="PrimaryFrameDataset", disable=quiet):
            assert issue in ISSUES

            raw_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))
            token_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_tokenized.json"))
            testset_ids = set(
                load_json(join(FRAMING_DATA_DIR, f"{issue}_test_sets.json"))[
                    "primary_frame"
                ]
            )

            for k, v in tqdm(raw_data.items(), disable=True):
                if split == "train" and k in testset_ids:
                    continue
                if split == "test" and k not in testset_ids:
                    continue
                if (
                    v["irrelevant"] == 1
                    or v["primary_frame"] == None
                    or v["primary_frame"] == 0
                ):
                    continue
                if k not in token_data:
                    continue

                self.all_data.append(
                    {
                        "tokens": token_data[k],
                        "label": PrimaryFrameDataset.get_frame_index(
                            v["primary_frame"]
                        ),
                        "text": v["text"],
                    }
                )

    def __len__(self) -> int:
        return len(self.all_data)

    def __getitem__(self, idx: int):
        sample = self.all_data[idx]
        return {
            "x": np.array(sample["tokens"]),
            "y": sample["label"],
        }

    @staticmethod
    def get_frame_index(frame_float: float):
        assert frame_float != 0
        return int(frame_float) - 1


if __name__ == "__main__":
    dataset = PrimaryFrameDataset(
        ISSUES,
        "test",
    )
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    for batch in dataloader:
        print(batch["x"].shape)
