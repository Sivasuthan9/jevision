"""Linear probes and readout recalibration for experiment 1."""
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def probe_accuracy(x_train, y_train, x_test, y_test, n_pca=256, n_jobs=3):
    """Test accuracy of a logistic-regression probe; PCA (fitted on the training half) above n_pca dimensions,
    regularisation C chosen by 3-fold cross-validation on the training half."""
    x_train, x_test = np.asarray(x_train, np.float32), np.asarray(x_test, np.float32)
    if x_train.shape[1] > n_pca:
        sc = StandardScaler().fit(x_train)
        pca = PCA(n_pca, random_state=0).fit(sc.transform(x_train))
        x_train, x_test = pca.transform(sc.transform(x_train)), pca.transform(sc.transform(x_test))
    clf = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                       {"logisticregression__C": [1e-2, 1e-1, 1]}, cv=3, n_jobs=n_jobs)
    clf.fit(x_train, y_train)
    return float((clf.predict(x_test) == y_test).mean())


def pixel_histogram(image_path):
    """64-bin colour histogram: a pixel-level control feature with no model involved."""
    a = np.asarray(Image.open(image_path).convert("RGB")).reshape(-1, 3) // 64
    h = np.bincount(a[:, 0] * 16 + a[:, 1] * 4 + a[:, 2], minlength=64).astype(float)
    return h / h.sum()


def recalibrated_readout_accuracy(logits_train, y_train, logits_test, y_test, cs=(0.01, 0.1, 1, 10)):
    """Accuracy of a logistic regression on the 8 option-letter logits (a linear recalibration of the readout).

    As in the pilot, the best C is taken on the test split, so this is an optimistic upper bound.
    """
    return max(float((LogisticRegression(C=c, max_iter=3000).fit(logits_train, y_train).predict(logits_test)
                      == y_test).mean()) for c in cs)


def yes_no_probe(features, y, train, test, layers=(8, 12, 16, 20, 24, 28, 32), n_pca=256, c=0.1):
    """Per-layer probe on answer-position states for the yes/no random-code task.

    Returns {layer: (test predictions, test P(yes))}.
    """
    out = {}
    for layer in layers:
        x = features[:, layer].float().numpy()
        sc = StandardScaler().fit(x[train])
        pca = PCA(n_pca, random_state=0).fit(sc.transform(x[train]))
        z = lambda a: pca.transform(sc.transform(a))  # noqa: E731
        clf = LogisticRegression(C=c, max_iter=3000).fit(z(x[train]), y[train])
        out[layer] = (clf.predict(z(x[test])), clf.predict_proba(z(x[test]))[:, 1])
    return out


def auc(y, score):
    return float(roc_auc_score(y, score))
