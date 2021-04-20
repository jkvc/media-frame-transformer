import re
import sys
from collections import Counter
from dataclasses import dataclass
from os.path import exists, join
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from config import DATA_DIR, FRAMING_DATA_DIR, ISSUES
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from numpy.random import f
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from media_frame_transformer.experiment_config import VOCAB_SIZE
from media_frame_transformer.utils import load_json

INPUT_N_TOKEN = 512
PAD_TOK_IDX = 1


@dataclass
class TextSample:
    text: str
    code: float
    issue: str
    subframes: Set[int]
    weight: float = 1


def pad_encoded(x):
    return x + ([PAD_TOK_IDX] * (INPUT_N_TOKEN - len(x)))


def clean_text(text):
    lines = text.split("\n\n")
    lines = lines[3:]  # first 3 lines are id, "PRIMARY", title
    text = "\n".join(lines)
    return text


def load_all_primary_frame_samples(issues: List[str]) -> List[TextSample]:
    samples = []
    for issue in tqdm(issues):
        train_set_ids = load_json(join(FRAMING_DATA_DIR, f"{issue}_train_sets.json"))[
            "primary_frame"
        ]
        raw_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))
        articleid2subframes = load_json(join(DATA_DIR, "subframes", f"{issue}.json"))

        for id in train_set_ids:
            item = raw_data[id]
            samples.append(
                TextSample(
                    text=clean_text(raw_data[id]["text"]),
                    code=raw_data[id]["primary_frame"],
                    issue=issue,
                    subframes=set(articleid2subframes[id]),
                    weight=1,
                )
            )
    return samples


def load_kfold_primary_frame_samples(
    issues: List[str], k: int
) -> List[Dict[str, List[TextSample]]]:
    for issue in issues:
        assert exists(
            join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json")
        ), f"{issue}_{k}_folds.json does not exist, run create_kfold first"

    fold2split2samples = [{"train": [], "valid": []} for _ in range(k)]

    for issue in tqdm(issues):
        raw_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_labeled.json"))
        kfold_data = load_json(join(FRAMING_DATA_DIR, f"{issue}_{k}_folds.json"))
        articleid2subframes = load_json(join(DATA_DIR, "subframes", f"{issue}.json"))

        for ki, fold in enumerate(kfold_data["primary_frame"]):
            for split in ["train", "valid"]:
                for id in fold[split]:
                    item = raw_data[id]
                    fold2split2samples[ki][split].append(
                        TextSample(
                            text=clean_text(item["text"]),
                            code=item["primary_frame"],
                            issue=issue,
                            subframes=set(articleid2subframes[id]),
                            weight=1,
                        )
                    )
    return fold2split2samples


def frame_code_to_idx(frame_float: float) -> int:
    # see codes.json, non null frames are [1.?, 15.?], map them to [0, 14]
    assert frame_float != 0
    assert frame_float < 16
    return int(frame_float) - 1


def label_idx_to_frame_code(idx: int) -> float:
    return float(idx + 1)


CODES = None


def idx_to_frame_name(idx) -> str:
    global CODES
    if CODES == None:
        CODES = load_json(join(DATA_DIR, "framing_labeled", "codes.json"))
    return CODES[f"{idx+1}.0"]


def load_label_distributions():
    return {
        t: load_json(join(DATA_DIR, "distributions", f"{t}.json"))
        for t in ["primary", "secondary", "both"]
    }


STOPWORDS = stopwords.words("english")


def get_tokens(cleaned_text: str) -> List[str]:
    text = cleaned_text.lower()
    nopunc = re.sub(r"[^\w\s]", "", text)
    tokens = nopunc.split()
    tokens = [w for w in tokens if w not in STOPWORDS and not w.isdigit()]
    return tokens


class PrimaryFrameDataset(Dataset):
    def __init__(
        self,
        samples: List[TextSample],
        issue2props_override: Optional[Dict[str, np.ndarray]] = None,
        vocab=None,
    ):
        self.samples: List[TextSample] = samples
        self.tokenizer = None
        self.label_distributions = load_label_distributions()
        if issue2props_override is not None:
            self.label_distributions = {
                "primary": issue2props_override,
                "secondary": issue2props_override,
                "both": issue2props_override,
            }

        lemmeatizer = WordNetLemmatizer()
        self.all_lemmas = [
            [lemmeatizer.lemmatize(w) for w in get_tokens(sample.text)]
            for sample in tqdm(samples, "lemmatize")
        ]
        if vocab is None:
            # build vocab
            word2count = Counter()
            for lemmas in self.all_lemmas:
                word2count.update(lemmas)
            mostcommon = word2count.most_common(VOCAB_SIZE)
            self.vocab = [w for w, c in mostcommon]
        else:
            print("using given vocab")
            self.vocab = vocab

        self.vocab_size = len(self.vocab)
        self.lemma2idx = {w: i for i, w in enumerate(self.vocab)}

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

        primary_frame_idx = frame_code_to_idx(sample.code)
        primary_frame_vec = _get_vector_from_idxs([primary_frame_idx])
        secondary_frame_vec = _get_vector_from_idxs(list(sample.subframes))
        both_frame_vec = _get_vector_from_idxs(
            list(sample.subframes.union({primary_frame_idx}))
        )

        bow_feat = np.zeros((self.vocab_size,))
        for lemma in self.all_lemmas[idx]:
            if lemma in self.lemma2idx:
                bow_feat[self.lemma2idx[lemma]] = 1

        return {
            "x": x,
            "weight": sample.weight,
            "primary_frame_idx": primary_frame_idx,
            "primary_frame_vec": primary_frame_vec,
            "secondary_frame_vec": secondary_frame_vec,
            "both_frame_vec": both_frame_vec,
            "primary_frame_distr": np.array(
                self.label_distributions["primary"][sample.issue]
            ),
            "secondary_frame_distr": np.array(
                self.label_distributions["secondary"][sample.issue]
            ),
            "both_frame_distr": np.array(
                self.label_distributions["both"][sample.issue]
            ),
            "bow": bow_feat,
        }


def _get_vector_from_idxs(idxs):
    vec = np.zeros((15,))
    vec[idxs] = 1
    return vec


def get_kfold_primary_frames_datasets(
    issues: List[str], k: int
) -> List[Dict[str, List[PrimaryFrameDataset]]]:
    fold2split2samples = load_kfold_primary_frame_samples(issues, k)
    return fold2split2samples_to_datasets(fold2split2samples)


def fold2split2samples_to_datasets(fold2split2samples):
    fold2split2datasets = [
        {
            split_name: PrimaryFrameDataset(split_samples)
            for split_name, split_samples in split2samples.items()
        }
        for split2samples in fold2split2samples
    ]
    return fold2split2datasets


if __name__ == "__main__":
    fold2split2samples = load_kfold_primary_frame_samples(["climate", "tobacco"], 8)
    for ki, split2samples in enumerate(fold2split2samples):
        train_samples = split2samples["train"]
        train_ds = PrimaryFrameDataset(train_samples)
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=30, num_workers=2)
        for batch in train_loader:
            print(batch)
            break
