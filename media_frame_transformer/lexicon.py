import re
from collections import Counter
from os import makedirs
from os.path import join
from typing import Dict, List, Tuple

import numpy as np
import torch
from config import VOCAB_SIZE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.optim import AdamW
from tqdm import tqdm, trange

import media_frame_transformer.models_lexicon  # noqa
from media_frame_transformer.dataset import (
    PRIMARY_FRAME_NAMES,
    get_primary_frame_labelprops_full_split,
    primary_frame_code_to_cidx,
)
from media_frame_transformer.learning import calc_f1
from media_frame_transformer.models import get_model, get_model_names
from media_frame_transformer.text_samples import TextSample
from media_frame_transformer.utils import DEVICE, save_json, write_str_list_as_txt

STOPWORDS = stopwords.words("english")


def get_tokens(cleaned_text: str) -> List[str]:
    text = cleaned_text.lower()
    nopunc = re.sub(r"[^\w\s]", "", text)
    tokens = nopunc.split()
    tokens = [w for w in tokens if w not in STOPWORDS and not w.isdigit()]
    return tokens


def build_lemma_vocab(
    samples: List[TextSample],
) -> Tuple[List[str], Dict[str, int], List[List[str]]]:
    lemmeatizer = WordNetLemmatizer()
    all_lemmas = [
        [lemmeatizer.lemmatize(w) for w in get_tokens(sample.text)]
        for sample in tqdm(samples)
    ]
    word2count = Counter()
    for lemmas in all_lemmas:
        word2count.update(lemmas)
    vocab = [w for w, c in word2count.most_common(VOCAB_SIZE)]
    word2idx = {w: i for i, w in enumerate(vocab)}

    return vocab, word2idx, all_lemmas


def build_bow_xys(
    samples: List[TextSample],
    all_lemmas: List[List[str]],
    word2idx: Dict[str, int],
) -> Tuple[np.array, np.array]:
    X = np.zeros((len(samples), len(word2idx)))
    y = np.zeros((len(samples),))

    for i, sample in enumerate(tqdm(samples)):
        lemmas = all_lemmas[i]
        for w in lemmas:
            if w in word2idx:
                X[i, word2idx[w]] += 1
        y[i] = primary_frame_code_to_cidx(sample.code)

    return X, y


def train_lexicon_model(arch, samples):
    vocab, word2idx, all_lemmas = build_lemma_vocab(samples)
    X, y = build_bow_xys(samples, all_lemmas, word2idx)

    issue2labelprops = get_primary_frame_labelprops_full_split("train")
    labelprops = np.array([issue2labelprops[s.issue] for s in samples])
    train_batch = {
        "x": torch.Tensor(X),
        "y": torch.LongTensor(y),
        "label_distribution": torch.FloatTensor(labelprops),
    }

    model = get_model(arch).to(DEVICE)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.5)

    for e in trange(3000):
        optimizer.zero_grad()
        outputs = model(train_batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

    train_outputs = model(train_batch)
    f1, precision, recall = calc_f1(
        train_outputs["logits"].detach().cpu().numpy(),
        train_outputs["labels"].detach().cpu().numpy(),
    )
    metrics = {"f1": f1, "precision": precision, "recall": recall}

    return vocab, model, metrics


def run_lexicon_experiment(arch, samples, logdir):
    assert arch in get_model_names()

    makedirs(logdir, exist_ok=True)

    vocab, model, metrics = train_lexicon_model(arch, samples)

    write_str_list_as_txt(vocab, join(logdir, "vocab.txt"))
    torch.save(model, join(logdir, "model.pth"))
    save_json(metrics, join(logdir, "leaf_metrics.json"))

    df = model.get_weighted_lexicon(vocab, PRIMARY_FRAME_NAMES)
    df.to_csv(join(logdir, "lexicon.csv"), index=False)
