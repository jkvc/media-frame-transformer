import re
import sys
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from numpy.random import f
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from media_frame_transformer.dataset import frame_code_to_idx, idx_to_frame_name

STOPWORDS = stopwords.words("english")
TOP_N_WORD = 1000


def get_tokens(cleaned_text: str) -> List[str]:
    text = cleaned_text.lower()
    nopunc = re.sub(r"[^\w\s]", "", text)
    tokens = nopunc.split()
    tokens = [w for w in tokens if w not in STOPWORDS and not w.isdigit()]
    return tokens


def build_lexicon(samples):
    lemmeatizer = WordNetLemmatizer()
    all_lemmas = [
        [lemmeatizer.lemmatize(w) for w in get_tokens(sample.text)]
        for sample in tqdm(samples)
    ]

    word2count = Counter()
    for lemmas in all_lemmas:
        word2count.update(lemmas)
    vocab = [w for w, c in word2count.most_common(TOP_N_WORD)]
    word2idx = {w: i for i, w in enumerate(vocab)}

    X = np.zeros((len(samples), len(vocab)))
    y = np.zeros((len(samples),))
    for i, sample in enumerate(tqdm(samples)):
        lemmas = all_lemmas[i]
        for w in lemmas:
            if w in word2idx:
                X[i, word2idx[w]] += 1
        y[i] = frame_code_to_idx(sample.code)

    model = LogisticRegression(
        multi_class="multinomial",
        max_iter=5000,
    )
    model.fit(X, y)

    preds = model.predict(X)
    print((preds == y).sum() / len(y))

    df = pd.DataFrame()
    df["word"] = vocab
    for c in range(15):
        if c >= len(model.coef_):
            continue
        df[f"{c}: {idx_to_frame_name(c)}"] = model.coef_[c]
    return df
