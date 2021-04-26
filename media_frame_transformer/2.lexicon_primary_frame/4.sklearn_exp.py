import sys
from os import makedirs
from os.path import join

import pandas as pd
from config import ISSUES, LEX_DIR
from media_frame_transformer.dataset import PRIMARY_FRAME_NAMES
from media_frame_transformer.eval import reduce_and_save_metrics
from media_frame_transformer.lexicon import (
    build_bow_xys,
    build_lemma_vocab,
    eval_lexicon_model,
    lemmatize,
    run_lexicon_experiment,
)
from media_frame_transformer.text_samples import load_all_text_samples
from media_frame_transformer.utils import save_json
from numpy.random import f
from sklearn.linear_model import LogisticRegression


def _get_samples(holdout_issue):
    if holdout_issue == "none":
        train_samples = load_all_text_samples(ISSUES, "train", "primary_frame")
        valid_samples = train_samples
    else:
        train_issues = [i for i in ISSUES if i != holdout_issue]
        train_samples = load_all_text_samples(train_issues, "train", "primary_frame")
        valid_samples = load_all_text_samples([holdout_issue], "train", "primary_frame")
    return train_samples, valid_samples


if __name__ == "__main__":
    for holdout_issue in ["none"] + ISSUES:
        print(">> holdout", holdout_issue)

        train_samples, valid_samples = _get_samples(holdout_issue)

        vocab, all_lemmas = build_lemma_vocab(train_samples)
        train_x, train_y = build_bow_xys(train_samples, all_lemmas, vocab)
        valid_x, valid_y = build_bow_xys(valid_samples, lemmatize(valid_samples), vocab)

        logreg = LogisticRegression(
            solver="lbfgs", multi_class="ovr", penalty="l2", max_iter=5000
        )
        logreg.fit(train_x, train_y)
        print(logreg.score(train_x, train_y), logreg.score(valid_x, valid_y))

        weights = logreg.coef_
        df = pd.DataFrame()
        df["word"] = vocab
        for c in range(15):
            df[PRIMARY_FRAME_NAMES[c]] = weights[c]

        makedirs(join(LEX_DIR, "4.sklearn_exp"), exist_ok=True)
        df.to_csv(
            join(LEX_DIR, "4.sklearn_exp", f"holdout_{holdout_issue}.csv"), index=False
        )
