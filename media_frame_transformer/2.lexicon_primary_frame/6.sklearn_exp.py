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
from media_frame_transformer.lexicon import build_bow_xys, build_lemma_vocab, lemmatize
from media_frame_transformer.text_samples import load_all_text_samples
from numpy.random import f
from scipy import optimize
from scipy.special._logsumexp import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing._label import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot, squared_norm
from sklearn.utils.multiclass import check_classification_targets


class LOGREG(LogisticRegression):
    def fit(self, X, y, log_labelprops):
        assert len(log_labelprops == len(X))

        _dtype = np.float64
        solver = "lbfgs"
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=_dtype,
            order="C",
            accept_large_sparse=solver != "liblinear",
        )
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        assert log_labelprops.shape[1] == len(self.classes_)

        n_classes = len(self.classes_)
        C_ = self.C
        _, n_features = X.shape
        classes = np.unique(y)

        lbin = LabelBinarizer()
        Y_multi = lbin.fit_transform(y)
        target = Y_multi

        w0 = np.zeros((classes.size, n_features), order="F", dtype=X.dtype)

        def multinomial_loss_grad(w, X, Y, alpha, sample_weight):
            n_classes = Y.shape[1]
            w = w.reshape(n_classes, -1)

            p = safe_sparse_dot(X, w.T)
            # print(X.shape, w.shape, p.shape)

            p += log_labelprops
            p -= logsumexp(p, axis=1)[:, np.newaxis]
            loss = -(Y * p).sum()
            loss += 0.5 * alpha * squared_norm(w)
            p = np.exp(p, p)

            diff = p - Y
            grad = safe_sparse_dot(diff.T, X)
            grad += alpha * w
            return loss, grad.ravel()

        opt_res = optimize.minimize(
            multinomial_loss_grad,
            w0,
            method="L-BFGS-B",
            jac=True,
            args=(X, target, 1.0 / [C_], self.sample_weight),
            options={"gtol": self.tol, "maxiter": self.max_iter},
        )
        w0 = opt_res.x
        self.coef_ = w0.reshape(n_classes, -1)
        self.intercept_ = np.zeros(n_classes)


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
    issue2labelprops = get_primary_frame_labelprops_full_split("train")

    for holdout_issue in ISSUES + ["none"]:
        print(">> holdout", holdout_issue)

        train_samples, valid_samples = _get_samples(holdout_issue)
        # train_samples = valid_samples

        vocab, all_lemmas = build_lemma_vocab(train_samples)
        train_x, train_y = build_bow_xys(train_samples, all_lemmas, vocab)
        valid_x, valid_y = build_bow_xys(valid_samples, lemmatize(valid_samples), vocab)

        labelprops = np.array([issue2labelprops[s.issue] for s in train_samples])
        log_labelprops = np.log(labelprops)

        logreg = LOGREG(
            solver="lbfgs",
            multi_class="multinomial",
            penalty="l2",
            max_iter=5000,
            C=0.025,
            fit_intercept=False,
        )
        logreg.fit(
            train_x,
            train_y,
            log_labelprops,
        )
        print("trainacc", logreg.score(train_x, train_y))

        preds = np.argmax(valid_x @ logreg.coef_.T + logreg.intercept_, axis=1)
        print("unbiased validacc", (preds == valid_y).sum() / len(valid_y))

        labelprops = issue2labelprops[holdout_issue]
        labelprops = np.log(labelprops)
        intercepts = labelprops
        preds = np.argmax(valid_x @ logreg.coef_.T + intercepts, axis=1)
        print("log biased validacc", (preds == valid_y).sum() / len(valid_y))

        # labelprops = issue2labelprops[holdout_issue]
        # labelprops = np.log(labelprops)
        # labelprops = (labelprops - labelprops.mean()) / labelprops.std()
        # intercepts = (labelprops * logreg.intercept_.std()) + logreg.intercept_.mean()
        # preds = np.argmax(valid_x @ logreg.coef_.T + intercepts, axis=1)
        # print("logz biased validacc", (preds == valid_y).sum() / len(valid_y))

        # print(logreg.coef_.shape, valid_x.shape, logreg.intercept_.shape)
        # logits = (logreg.coef_ @ valid_x.T).T + logreg.intercept_
        # print(logreg.intercept_)
        # props = np.array(
        #     [issue2labelprops[i] for i in ISSUES if i != holdout_issue]
        # ).mean(axis=0)
        # print(props)
        # print(logreg.coef_.mean(axis=1))
        # print(np.log(props))
        # print(np.log(props) - np.log(props).mean())

        weights = logreg.coef_
        df = pd.DataFrame()
        df["word"] = vocab
        for c in range(15):
            df[PRIMARY_FRAME_NAMES[c]] = weights[c]

        makedirs(join(LEX_DIR, "6.sklearn_exp"), exist_ok=True)
        df.to_csv(
            join(LEX_DIR, "6.sklearn_exp", f"holdout_{holdout_issue}.csv"), index=False
        )
