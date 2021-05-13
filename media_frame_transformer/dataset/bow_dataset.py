import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from media_frame_transformer.datadef.zoo import DatasetDefinition
from media_frame_transformer.dataset.data_sample import DataSample
from media_frame_transformer.utils import DEVICE
from nltk.corpus import stopwords
from tqdm import tqdm

STOPWORDS = stopwords.words("english")


def get_tokens(cleaned_text: str) -> List[str]:
    text = cleaned_text.lower()
    nopunc = re.sub(r"[^\w\s]", "", text)
    tokens = nopunc.split()
    tokens = [w for w in tokens if w not in STOPWORDS and not w.isdigit()]
    return tokens


def get_all_tokens(samples: List[DataSample]) -> List[List[str]]:
    all_tokens = [get_tokens(sample.text) for sample in tqdm(samples)]
    return all_tokens


def build_vocab(
    samples: List[DataSample], vocab_size: int
) -> Tuple[List[str], Dict[str, int], List[List[str]]]:
    all_tokens = get_all_tokens(samples)
    word2count = Counter()
    for tokens in all_tokens:
        word2count.update(tokens)
    vocab = [w for w, c in word2count.most_common(vocab_size)]

    return vocab, all_tokens


def build_bow_full_batch(
    samples: List[DataSample],
    datadef: DatasetDefinition,
    all_tokens: List[List[str]],
    vocab: List[str],
    use_source_individual_norm: bool,
    labelprop_split: str,
):
    word2idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(samples), len(word2idx)))
    y = np.zeros((len(samples),))

    for i, sample in enumerate((samples)):
        tokens = all_tokens[i]
        for w in tokens:
            if w in word2idx:
                X[i, word2idx[w]] = 1
        y[i] = sample.y_idx

    # normalize word freq within each issue
    if use_source_individual_norm:
        source_idxs = set(sample.source_idx for sample in samples)
        for source_idx in source_idxs:
            idxs = [
                i for i, sample in enumerate(samples) if sample.source_idx == source_idx
            ]
            if len(idxs) == 0:
                continue
            X[idxs] -= X[idxs].mean(axis=0)

    source2labelprops = datadef.load_labelprops_func(labelprop_split)
    labelprops = torch.FloatTensor([source2labelprops[s.source_name] for s in samples])

    source_idx = torch.LongTensor([s.source_idx for s in samples])

    batch = {
        "x": torch.FloatTensor(X),
        "y": torch.LongTensor(y),
        "labelprops": labelprops,
        "source_idx": source_idx.to(torch.long),
    }
    for k in batch:
        batch[k] = batch[k].to(DEVICE)
    return batch
