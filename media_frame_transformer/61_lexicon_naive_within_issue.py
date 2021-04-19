import re
import sys
from collections import Counter, defaultdict
from os import makedirs, mkdir
from os.path import exists, join
from pprint import pprint
from random import Random, shuffle
from typing import List

import numpy as np
import pandas as pd
from config import ISSUES, LEX_DIR, MODELS_DIR
from matplotlib.pyplot import xlabel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from numpy.random import f
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from media_frame_transformer.dataset import (
    PrimaryFrameDataset,
    fold2split2samples_to_datasets,
    frame_code_to_idx,
    idx_to_frame_name,
    load_all_primary_frame_samples,
    load_kfold_primary_frame_samples,
)
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.experiment_config import (
    ARCH,
    BATCHSIZE,
    DATASET_SIZES,
    FOLDS_TO_RUN,
    KFOLD,
)
from media_frame_transformer.experiments import run_experiments
from media_frame_transformer.learning import train
from media_frame_transformer.utils import (
    load_json,
    mkdir_overwrite,
    write_str_list_as_txt,
)
from media_frame_transformer.viualization import (
    plot_series_w_labels,
    visualize_num_sample_num_epoch,
)

STOPWORDS = stopwords.words("english")
TOP_N_WORD = 100


def get_tokens(cleaned_text: str) -> List[str]:
    text = cleaned_text.lower()
    nopunc = re.sub(r"[^\w\s]", "", text)
    tokens = nopunc.split()
    tokens = [w for w in tokens if w not in STOPWORDS and not w.isdigit()]
    return tokens


if __name__ == "__main__":
    lemmeatizer = WordNetLemmatizer()

    makedirs(join(LEX_DIR, "61.naive.within_issue"), exist_ok=True)

    for issue in ISSUES:
        all_samples = load_all_primary_frame_samples([issue])
        word2count = Counter()
        for sample in all_samples:
            word2count.update(
                [lemmeatizer.lemmatize(w) for w in get_tokens(sample.text)]
            )
        vocab = [w for w, c in word2count.most_common(TOP_N_WORD)]
        word2idx = {w: i for i, w in enumerate(vocab)}

        X = np.zeros((len(all_samples), len(vocab)))
        y = np.zeros((len(all_samples),))
        for i, sample in enumerate(tqdm(all_samples)):
            words = [lemmeatizer.lemmatize(w) for w in get_tokens(sample.text)]
            for w in words:
                if w in word2idx:
                    X[i, word2idx[w]] += 1
            y[i] = frame_code_to_idx(sample.code)

        model = LogisticRegression(multi_class="ovr", max_iter=5000)
        model.fit(X, y)

        df = pd.DataFrame()
        df["word"] = vocab
        for c in range(15):
            if c >= len(model.coef_):
                continue
            df[f"{c}: {idx_to_frame_name(c)}"] = model.coef_[c]

        df.to_csv(join(LEX_DIR, "61.naive.within_issue", f"{issue}.csv"))
