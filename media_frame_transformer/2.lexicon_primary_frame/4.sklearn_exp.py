import sys
from os import makedirs
from os.path import join

import numpy as np
import pandas as pd
from config import ISSUES, LEX_DIR
from media_frame_transformer.dataset import (
    PRIMARY_FRAME_NAMES,
    get_primary_frame_labelprops_full_split,
)
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

    issue2logreg = {}
    vocab, all_lemmas = build_lemma_vocab(
        load_all_text_samples(ISSUES, "train", "primary_frame")
    )

    for issue in ISSUES:
        print(">>", issue)

        # train_samples, valid_samples = _get_samples(holdout_issue)
        train_samples = load_all_text_samples([issue], "train", "primary_frame")
        train_x, train_y = build_bow_xys(train_samples, all_lemmas, vocab)
        # valid_x, valid_y = build_bow_xys(valid_samples, lemmatize(valid_samples), vocab)

        logreg = LogisticRegression(
            solver="lbfgs",
            multi_class="multinomial",
            penalty="l2",
            max_iter=5000,
            C=0.025,
        )
        logreg.fit(train_x, train_y)

        issue2logreg[issue] = logreg

        weights = logreg.coef_
        df = pd.DataFrame()
        df["word"] = vocab
        for c in range(15):
            df[PRIMARY_FRAME_NAMES[c]] = weights[c]

        makedirs(join(LEX_DIR, "4.sklearn_exp"), exist_ok=True)
        df.to_csv(join(LEX_DIR, "4.sklearn_exp", f"{issue}.csv"), index=False)

    issue2labelprops = get_primary_frame_labelprops_full_split("train")
    for holdout in ISSUES:
        w = np.array([issue2logreg[i].coef_ for i in ISSUES if i != holdout]).sum(
            axis=0
        )
        b = np.log(issue2labelprops[holdout])

        valid_samples = load_all_text_samples([holdout], "train", "primary_frame")
        valid_x, valid_y = build_bow_xys(valid_samples, lemmatize(valid_samples), vocab)
        preds = np.argmax(valid_x @ w.T + b, axis=1)
        print((preds == valid_y).sum() / len(valid_y))

        # print("trainacc", logreg.score(train_x, train_y))
        # print("unbiased validacc", logreg.score(valid_x, valid_y))

        # preds = np.argmax(valid_x @ logreg.coef_.T + logreg.intercept_, axis=1)
        # print("unbiased validacc", (preds == valid_y).sum() / len(valid_y))

        # print(logreg.intercept_)
        # print(logreg.intercept_.mean())
        # # print(np.log(logreg.intercept_))

        # issue2labelprops = get_primary_frame_labelprops_full_split("train")
        # labelprops = issue2labelprops[issue]
        # print(labelprops)
        # print(np.log(labelprops))

        # print(np.log(labelprops) - logreg.intercept_)

        # labelprops = (labelprops - labelprops.mean()) / labelprops.std()
        # intercepts = (labelprops * logreg.intercept_.std()) + logreg.intercept_.mean()
        # preds = np.argmax(valid_x @ logreg.coef_.T + intercepts, axis=1)
        # print("  biased validacc", (preds == valid_y).sum() / len(valid_y))

        # labelprops = issue2labelprops[holdout_issue]
        # labelprops = np.log(labelprops)
        # intercepts = labelprops
        # preds = np.argmax(valid_x @ logreg.coef_.T + intercepts, axis=1)
        # print("log biased validacc", (preds == valid_y).sum() / len(valid_y))

        # labelprops = issue2labelprops[holdout_issue]
        # labelprops = np.log(labelprops)
        # labelprops = (labelprops - labelprops.mean()) / labelprops.std()
        # intercepts = (labelprops * logreg.intercept_.std()) + logreg.intercept_.mean()
        # preds = np.argmax(valid_x @ logreg.coef_.T + intercepts, axis=1)
        # print("logz biased validacc", (preds == valid_y).sum() / len(valid_y))

        # # print(logreg.coef_.shape, valid_x.shape, logreg.intercept_.shape)
        # # logits = (logreg.coef_ @ valid_x.T).T + logreg.intercept_
        # # print(logreg.intercept_)
        # # props = np.array(
        # #     [issue2labelprops[i] for i in ISSUES if i != holdout_issue]
        # # ).mean(axis=0)
        # # print(props)
        # # print(logreg.coef_.mean(axis=1))
        # # print(np.log(props))
        # # print(np.log(props) - np.log(props).mean())
