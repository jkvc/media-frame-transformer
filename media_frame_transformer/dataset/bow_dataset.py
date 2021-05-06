import re
from collections import Counter
from os import makedirs
from os.path import join
from typing import Dict, List, Tuple

import numpy as np
import torch
from media_frame_transformer.dataset.common import get_labelprops_full_split
from media_frame_transformer.dataset.data_sample import DataSample
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
    all_tokens: List[List[str]],
    vocab: List[str],
    use_source_individual_norm: bool,
    labelprop_dir: str,
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

    source2labelprops = get_labelprops_full_split(labelprop_dir, labelprop_split)
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


def train_lexicon_model(
    model,
    train_samples,
    vocab_size,
    use_source_individual_norm,
    labelprop_dir,
    labelprop_split,
):
    vocab, all_tokens = build_vocab(train_samples, vocab_size)
    batch = build_bow_full_batch(
        train_samples,
        all_tokens,
        vocab,
        use_source_individual_norm,
        labelprop_dir,
        labelprop_split,
    )

    optimizer = SGD(model.parameters(), lr=1e-1, weight_decay=0)

    model.train()
    for e in trange(5000):
        optimizer.zero_grad()
        outputs = model(batch)

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

    train_outputs = model(batch)
    f1, precision, recall = calc_f1(
        train_outputs["logits"].detach().cpu().numpy(),
        train_outputs["labels"].detach().cpu().numpy(),
    )
    metrics = {"train_f1": f1, "train_precision": precision, "train_recall": recall}

    return vocab, model, metrics


def eval_lexicon_model(
    model,
    valid_samples,
    vocab,
    use_source_individual_norm,
    labelprop_dir,
    labelprop_split,
):
    batch = build_bow_full_batch(
        valid_samples,
        get_all_tokens(valid_samples),
        vocab,
        use_source_individual_norm,
        labelprop_dir,
        labelprop_split,
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


def run_lexicon_experiment(
    config,
    train_samples,
    valid_samples,
    vocab_size,
    logdir,
    source_names,
    labelprop_dir,
    train_labelprop_split,
    valid_labelprop_split,
):
    model = get_model(config).to(DEVICE)
    makedirs(logdir, exist_ok=True)

    use_source_individual_norm = config["use_source_individual_norm"]

    vocab, model, train_metrics = train_lexicon_model(
        model,
        train_samples,
        vocab_size,
        use_source_individual_norm,
        labelprop_dir,
        train_labelprop_split,
    )
    valid_metrics = eval_lexicon_model(
        model,
        valid_samples,
        vocab,
        use_source_individual_norm,
        labelprop_dir,
        valid_labelprop_split,
    )

    write_str_list_as_txt(vocab, join(logdir, "vocab.txt"))
    torch.save(model, join(logdir, "model.pth"))

    metrics = {}
    metrics.update(train_metrics)
    metrics.update(valid_metrics)
    print_metrics(metrics)
    save_json(metrics, join(logdir, "leaf_metrics.json"))

    df = model.get_weighted_lexicon(vocab, source_names)
    df.to_csv(join(logdir, "lexicon.csv"), index=False)

    return vocab, model, metrics, df
