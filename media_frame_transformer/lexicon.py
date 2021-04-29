import re
import sys
from collections import Counter
from os import makedirs
from os.path import join
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from config import ISSUES, LEX_DIR, N_CLASSES, VOCAB_SIZE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from media_frame_transformer.dataset import (
    PRIMARY_FRAME_NAMES,
    get_primary_frame_labelprops_full_split,
    primary_frame_code_to_cidx,
)
from media_frame_transformer.learning import _print_metrics
from media_frame_transformer.logreg import MultinomialLogisticRegressionWithLabelprops
from media_frame_transformer.text_samples import TextSample
from media_frame_transformer.utils import DEVICE, save_json, write_str_list_as_txt

STOPWORDS = stopwords.words("english")


def get_tokens(cleaned_text: str) -> List[str]:
    text = cleaned_text.lower()
    nopunc = re.sub(r"[^\w\s]", "", text)
    tokens = nopunc.split()
    tokens = [w for w in tokens if w not in STOPWORDS and not w.isdigit()]
    return tokens


def lemmatize(samples: List[TextSample]) -> List[List[str]]:
    lemmeatizer = WordNetLemmatizer()
    all_lemmas = [
        [lemmeatizer.lemmatize(w) for w in get_tokens(sample.text)]
        for sample in tqdm(samples)
    ]
    return all_lemmas


def build_lemma_vocab(
    samples: List[TextSample], vocab_size: int = VOCAB_SIZE
) -> Tuple[List[str], Dict[str, int], List[List[str]]]:
    all_lemmas = lemmatize(samples)
    word2count = Counter()
    for lemmas in all_lemmas:
        word2count.update(lemmas)
    vocab = [w for w, c in word2count.most_common(vocab_size)]

    return vocab, all_lemmas


def build_bow_xys(
    samples: List[TextSample],
    all_lemmas: List[List[str]],
    vocab: List[str],
    append_artificial_samples: bool = False,
) -> Tuple[np.array, np.array]:
    issue2labelprops = get_primary_frame_labelprops_full_split("train")

    word2idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(samples), len(word2idx)))
    y = np.zeros((len(samples),))
    labelprops = np.array([issue2labelprops[s.issue] for s in samples])

    for i, sample in enumerate(tqdm(samples)):
        lemmas = all_lemmas[i]
        for w in lemmas:
            if w in word2idx:
                X[i, word2idx[w]] += 1
        y[i] = primary_frame_code_to_cidx(sample.code)

    if append_artificial_samples:
        X = np.append(X, np.zeros((N_CLASSES, len(word2idx))), axis=0)
        y = np.append(y, np.arange(N_CLASSES))
        labelprops = np.append(
            labelprops, np.ones((N_CLASSES, N_CLASSES)) * (1 / N_CLASSES), axis=0
        )

    return X, y, labelprops


LEXICON_ARCHS = ["multinomial", "multinomial+dev"]


def run_lexicon_experiment(arch, C, train_samples, valid_samples, logdir):
    assert arch in LEXICON_ARCHS
    vocab, train_lemmas = build_lemma_vocab(train_samples)

    trainx, trainy, trainlabelprops = build_bow_xys(
        samples=train_samples,
        all_lemmas=train_lemmas,
        vocab=vocab,
        append_artificial_samples=True,
    )
    validx, validy, validlabelprops = build_bow_xys(
        samples=valid_samples,
        all_lemmas=lemmatize(valid_samples),
        vocab=vocab,
    )

    # fit and eval

    if arch == "multinomial":
        logreg = LogisticRegression(
            penalty="l2",
            C=C,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=5000,
            multi_class="multinomial",
        )
        logreg.fit(trainx, trainy)
        train_acc = logreg.score(trainx, trainy)
        valid_acc = logreg.score(validx, validy)
    elif arch == "multinomial+dev":
        logreg = MultinomialLogisticRegressionWithLabelprops(
            penalty="l2",
            C=C,
            fit_intercept=False,
            solver="lbfgs",
            max_iter=5000,
            multi_class="multinomial",
        )
        logreg.fit(trainx, trainy, trainlabelprops)
        train_acc = logreg.score(trainx, trainy, trainlabelprops)
        valid_acc = logreg.score(validx, validy, validlabelprops)

    makedirs(logdir, exist_ok=True)
    metrics = {"train_acc": train_acc, "valid_acc": valid_acc}
    _print_metrics(metrics)
    save_json(metrics, join(logdir, "leaf_metrics.json"))

    w = logreg.coef_
    df = pd.DataFrame()
    df["word"] = vocab
    for c in range(N_CLASSES):
        df[PRIMARY_FRAME_NAMES[c]] = w[c]
    df.to_csv(join(logdir, "lexicon.csv"), index=False)

    write_str_list_as_txt(vocab, join(logdir, "vocab.txt"))

    return vocab, metrics
