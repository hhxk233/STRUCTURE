import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def _to_2d(X: np.ndarray) -> np.ndarray:
    if X.shape[1] <= 2:
        return X[:, :2]
    return PCA(n_components=2).fit_transform(X)


def embedding_plot(X, y, label_dict=None, return_figure=False):
    X = _to_2d(np.asarray(X))
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(6, 5))
    for label in np.unique(y):
        mask = y == label
        name = label_dict.get(label, str(label)) if label_dict else str(label)
        ax.scatter(X[mask, 0], X[mask, 1], s=6, label=name)
    ax.legend(markerscale=2, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig if return_figure else ax


def embedding_plot_w_markers(
    X,
    y,
    text_X=None,
    text_y=None,
    label_dict=None,
):
    X = _to_2d(np.asarray(X))
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(6, 5))
    for label in np.unique(y):
        mask = y == label
        name = label_dict.get(label, str(label)) if label_dict else str(label)
        ax.scatter(X[mask, 0], X[mask, 1], s=6, label=name)

    if text_X is not None:
        text_X = _to_2d(np.asarray(text_X))
        ax.scatter(text_X[:, 0], text_X[:, 1], s=40, marker="x", c="black")
    ax.legend(markerscale=2, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
