import re
import sys
from collections import Counter, defaultdict
from os import makedirs
from os.path import exists, join
from pprint import pprint
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from config import ISSUES, LEX_DIR, MODELS_DIR
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from numpy.random import f
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from media_frame_transformer import models
from media_frame_transformer.dataset import (
    PrimaryFrameDataset,
    frame_code_to_idx,
    idx_to_frame_name,
    load_all_primary_frame_samples,
)
from media_frame_transformer.experiment_config import VOCAB_SIZE
from media_frame_transformer.learning import _calc_f1, train
from media_frame_transformer.lexicon import build_lexicon
from media_frame_transformer.models import get_model
from media_frame_transformer.utils import (
    DEVICE,
    mkdir_overwrite,
    save_json,
    write_str_list_as_txt,
)

_arch = sys.argv[1]


STOPWORDS = stopwords.words("english")


def get_tokens(cleaned_text: str) -> List[str]:
    text = cleaned_text.lower()
    nopunc = re.sub(r"[^\w\s]", "", text)
    tokens = nopunc.split()
    tokens = [w for w in tokens if w not in STOPWORDS and not w.isdigit()]
    return tokens


if __name__ == "__main__":
    makedirs(join(LEX_DIR, f"63.temp.{_arch}"), exist_ok=True)

    lemmeatizer = WordNetLemmatizer()

    train_samples = load_all_primary_frame_samples(ISSUES)
    all_lemmas = [
        [lemmeatizer.lemmatize(w) for w in get_tokens(sample.text)]
        for sample in tqdm(train_samples)
    ]

    word2count = Counter()
    for lemmas in all_lemmas:
        word2count.update(lemmas)
    vocab = [w for w, c in word2count.most_common(VOCAB_SIZE)]
    word2idx = {w: i for i, w in enumerate(vocab)}

    X = np.zeros((len(train_samples), len(vocab)))
    y = np.zeros((len(train_samples), 15))
    # y = np.zeros((len(train_samples),))

    for i, sample in enumerate(tqdm(train_samples)):
        lemmas = all_lemmas[i]
        for w in lemmas:
            if w in word2idx:
                X[i, word2idx[w]] += 1
        y[i, frame_code_to_idx(sample.code)] = 1

    # print(sum(1 for lemmas, c in zip(all_lemmas, y) if "los" in lemmas and c == 0))
    # print(len(y))
    # print(sum(1 for lemmas, c in zip(all_lemmas, y) if "economy" in lemmas and c == 0))
    # print(len(y))

    train_batch = {
        "bow": torch.Tensor(X),
        "primary_frame_vec": torch.FloatTensor(y),
        # "primary_frame_idx": torch.LongTensor(y),
    }

    model = get_model(_arch).to(DEVICE)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.5)

    for e in trange(3000):
        optimizer.zero_grad()
        outputs = model(train_batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

    train_outputs = model(train_batch)
    f1, precision, recall = _calc_f1(
        train_outputs["logits"].detach().cpu().numpy(),
        train_outputs["labels"].detach().cpu().numpy(),
    )
    print("train f1", f1)

    df = pd.DataFrame()
    df["word"] = vocab
    df["count"] = [word2count[word] for word in vocab]
    weights = model.ff.weight.data.detach().cpu().numpy()
    for c in range(15):
        df[f"{c}: {idx_to_frame_name(c)}"] = weights[c]
    df.to_csv(
        join(LEX_DIR, f"63.temp.{_arch}", f"all.csv"),
        index=False,
    )
