import re
from collections import Counter
from os import makedirs
from os.path import join
from typing import Dict, List, Tuple

import numpy as np
import torch
from media_frame_transformer.dataset.framing.common import (
    get_primary_frame_labelprops_full_split,
)
from media_frame_transformer.dataset.framing.definition import (
    ISSUES,
    PRIMARY_FRAME_NAMES,
)
from media_frame_transformer.dataset.framing.samples import FramingDataSample
from media_frame_transformer.learning import calc_f1, print_metrics
from media_frame_transformer.model import get_model
from media_frame_transformer.utils import DEVICE, save_json, write_str_list_as_txt
from nltk.corpus import stopwords
from torch.optim import ASGD, SGD, Adam, AdamW
from tqdm import tqdm, trange

STOPWORDS = stopwords.words("english")


def get_tokens(cleaned_text: str) -> List[str]:
    text = cleaned_text.lower()
    nopunc = re.sub(r"[^\w\s]", "", text)
    tokens = nopunc.split()
    tokens = [w for w in tokens if w not in STOPWORDS and not w.isdigit()]
    return tokens


def get_all_tokens(samples: List[FramingDataSample]) -> List[List[str]]:
    all_tokens = [get_tokens(sample.text) for sample in tqdm(samples)]
    return all_tokens


def build_vocab(
    samples: List[FramingDataSample], vocab_size: int
) -> Tuple[List[str], Dict[str, int], List[List[str]]]:
    all_tokens = get_all_tokens(samples)
    word2count = Counter()
    for tokens in all_tokens:
        word2count.update(tokens)
    vocab = [w for w, c in word2count.most_common(vocab_size)]

    return vocab, all_tokens


def build_bow_full_batch(
    samples: List[FramingDataSample],
    all_tokens: List[List[str]],
    vocab: List[str],
    use_source_individual_norm: bool,
):
    word2idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(samples), len(word2idx)))
    y = np.zeros((len(samples),))

    for i, sample in enumerate((samples)):
        tokens = all_tokens[i]
        for w in tokens:
            if w in word2idx:
                X[i, word2idx[w]] = 1
        y[i] = sample.frame_idx

    # normalize word freq within each issue
    for issue in ISSUES:
        idxs = [i for i, sample in enumerate(samples) if sample.issue == issue]
        if len(idxs) == 0:
            continue
        X[idxs] -= X[idxs].mean(axis=0)

    issue2labelprops = get_primary_frame_labelprops_full_split("train")
    labelprops = torch.FloatTensor([issue2labelprops[s.issue] for s in samples])

    issue_idx = torch.LongTensor([s.issue_idx for s in samples])

    batch = {
        "x": torch.FloatTensor(X),
        "y": torch.LongTensor(y),
        "labelprops": labelprops,
        "source_idx": issue_idx.to(torch.long),
    }
    for k in batch:
        batch[k] = batch[k].to(DEVICE)
    return batch


def train_lexicon_model(model, train_samples, vocab_size, use_source_individual_norm):
    vocab, all_tokens = build_vocab(train_samples, vocab_size)
    batch = build_bow_full_batch(
        train_samples, all_tokens, vocab, use_source_individual_norm
    )

    tol = 0.00001

    optimizer = SGD(model.parameters(), lr=1e-1, weight_decay=0)

    prev_loss = float("inf")
    model.train()
    for e in trange(10000):
        optimizer.zero_grad()
        outputs = model(batch)

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        loss = loss.item()
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    train_outputs = model(batch)
    f1, precision, recall = calc_f1(
        train_outputs["logits"].detach().cpu().numpy(),
        train_outputs["labels"].detach().cpu().numpy(),
    )
    metrics = {"train_f1": f1, "train_precision": precision, "train_recall": recall}

    return vocab, model, metrics


def eval_lexicon_model(model, valid_samples, vocab, use_source_individual_norm):
    batch = build_bow_full_batch(
        valid_samples, get_all_tokens(valid_samples), vocab, use_source_individual_norm
    )

    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    f1, precision, recall = calc_f1(
        outputs["logits"].detach().cpu().numpy(),
        outputs["labels"].detach().cpu().numpy(),
    )

    metrics = {"valid_f1": f1, "valid_precision": precision, "valid_recall": recall}
    return metrics


def run_lexicon_experiment(config, train_samples, valid_samples, vocab_size, logdir):
    model = get_model(config).to(DEVICE)
    makedirs(logdir, exist_ok=True)

    use_source_individual_norm = config["use_source_individual_norm"]

    vocab, model, train_metrics = train_lexicon_model(
        model, train_samples, vocab_size, use_source_individual_norm
    )
    valid_metrics = eval_lexicon_model(
        model, valid_samples, vocab, use_source_individual_norm
    )

    write_str_list_as_txt(vocab, join(logdir, "vocab.txt"))
    torch.save(model, join(logdir, "model.pth"))

    metrics = {}
    metrics.update(train_metrics)
    metrics.update(valid_metrics)
    print_metrics(metrics)
    save_json(metrics, join(logdir, "leaf_metrics.json"))

    df = model.get_weighted_lexicon(vocab, PRIMARY_FRAME_NAMES)
    df.to_csv(join(logdir, "lexicon.csv"), index=False)

    return vocab, model, metrics, df
