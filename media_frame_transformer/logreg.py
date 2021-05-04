import numpy as np
from numpy.random import f
from scipy import optimize
from scipy.special._logsumexp import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing._label import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot, squared_norm
from sklearn.utils.multiclass import check_classification_targets


class MultinomialLogisticRegressionWithLabelprops(LogisticRegression):
    def fit(self, X, y, labelprops, rs):
        assert len(labelprops == len(X))
        log_labelprops = np.log(labelprops)

        assert self.solver == "lbfgs"

        _dtype = np.float64
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=_dtype,
            order="C",
            accept_large_sparse=True,
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

            p += log_labelprops
            p -= logsumexp(p, axis=1)[:, np.newaxis]
            loss = -(Y * p).sum()
            loss += 0.5 * alpha * squared_norm(w)
            p = np.exp(p, p)

            diff = p - Y
            grad = safe_sparse_dot(diff.T, X)
            grad += alpha / 2 * w
            grad += alpha / 2 * np.expand_dims(rs, axis=0) * w  # FIXME
            return loss, grad.ravel()

        opt_res = optimize.minimize(
            multinomial_loss_grad,
            w0,
            method="L-BFGS-B",
            jac=True,
            args=(X, target, 1.0 / C_, None),
            options={"gtol": self.tol, "maxiter": self.max_iter},
        )
        w0 = opt_res.x
        self.coef_ = w0.reshape(n_classes, -1)
        self.intercept_ = np.zeros(n_classes)

    def score(self, X, y, labelprops):
        assert len(labelprops == len(X))
        log_labelprops = np.log(labelprops)

        preds = np.argmax(X @ self.coef_.T + log_labelprops, axis=1)
        return (preds == y).sum() / len(y)
