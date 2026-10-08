"""
classical.py, four hypothesis classes, one dataset, four shapes of assumption.

The Evolution of Machine Learning and AI · Week 3 Lab
South East Technological University

Booklet cross-references:
    Chapter 3  linear and logistic regression
    Chapter 4  trees and forests
    Chapter 5  support vector machines and the kernel trick
    Chapter 6  the bias-variance trade-off

The point of this file is not that one model wins. It is that each model draws a
DIFFERENT SHAPE of boundary, because each encodes a different assumption about
how the world is put together. That is Week 2's second sense of "bias" and
Week 4's whole thesis, arriving early and in pictures.
"""

from __future__ import annotations

from pathlib import Path

import sys
import matplotlib
if "ipykernel" not in sys.modules:      # keep inline plots working in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SLATE, AMBER, RED, BLUE, GREY, GREEN = "#47546B", "#A96E08", "#9E3B2C", "#4E7291", "#8A93A5", "#2E7D4F"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 8.5,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold"})


def models(seed: int = 0) -> dict:
    """Each entry is (estimator, the assumption it encodes)."""
    return {
        "Logistic regression": (
            make_pipeline(StandardScaler(), LogisticRegression()),
            "the classes are separable by a straight line"),
        "Decision tree (depth 3)": (
            DecisionTreeClassifier(max_depth=3, random_state=seed),
            "the world splits on one feature at a time, at right angles"),
        "Decision tree (unlimited)": (
            DecisionTreeClassifier(random_state=seed),
            "same, but keep splitting until every leaf is pure"),
        "Random forest": (
            RandomForestClassifier(n_estimators=200, random_state=seed),
            "average many deliberately different trees"),
        "SVM (linear)": (
            make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
            "a straight line, placed to maximise the margin"),
        "SVM (RBF kernel)": (
            make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma="scale")),
            "a straight line, in a space we never actually visit"),
        "Neural net (8, 8)": (
            make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=4000,
                                        random_state=seed)),
            "compose simple bends until the boundary fits"),
    }


def load(n=600, noise=0.28, seed=0):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)


def compare(seed: int = 0) -> pd.DataFrame:
    """Train everything, report train and test accuracy side by side.

    Read the GAP between the two columns, not the test column alone. A large gap
    is the signature of a model that memorised rather than generalised, and it is
    the whole content of Chapter 6.
    """
    Xtr, Xte, ytr, yte = load(seed=seed)
    rows = []
    for name, (est, assumption) in models(seed).items():
        est.fit(Xtr, ytr)
        tr, te = est.score(Xtr, ytr), est.score(Xte, yte)
        rows.append({"model": name, "train": round(tr, 3), "test": round(te, 3),
                     "gap": round(tr - te, 3), "assumption encoded": assumption})
    return pd.DataFrame(rows)


def fig_boundaries(seed: int = 0):
    """The slide. Seven boundaries, seven assumptions, one dataset."""
    Xtr, Xte, ytr, yte = load(seed=seed)
    ms = models(seed)
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.0))
    axes = axes.ravel()

    x_min, x_max = Xtr[:, 0].min() - 0.6, Xtr[:, 0].max() + 0.6
    y_min, y_max = Xtr[:, 1].min() - 0.6, Xtr[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 320),
                         np.linspace(y_min, y_max, 320))
    grid = np.c_[xx.ravel(), yy.ravel()]

    axes[0].scatter(Xtr[:, 0], Xtr[:, 1], c=[SLATE if v == 0 else AMBER for v in ytr],
                    s=9, alpha=0.85)
    axes[0].set_title("the data\n(two interleaving moons)")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    for ax, (name, (est, _)) in zip(axes[1:], ms.items()):
        est.fit(Xtr, ytr)
        Z = est.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                    colors=[SLATE, AMBER], alpha=0.16)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[RED], linewidths=1.4)
        ax.scatter(Xtr[:, 0], Xtr[:, 1], c=[SLATE if v == 0 else AMBER for v in ytr],
                   s=6, alpha=0.7)
        ax.set_title(f"{name}\ntrain {est.score(Xtr, ytr):.2f}  ·  test {est.score(Xte, yte):.2f}")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("The same data. Seven assumptions about how the world is put together.",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Under- and over-fitting, on a curve you can see
# ---------------------------------------------------------------------------

def fig_fit(seed: int = 0):
    """Polynomial regression at three capacities: too little, right, far too much."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, 26))
    y_true = np.sin(2 * np.pi * x)
    y = y_true + rng.normal(0, 0.22, len(x))
    xs = np.linspace(0, 1, 400)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    rows = []
    for ax, deg, label in zip(axes, (1, 4, 18),
                              ("degree 1 · UNDERFIT", "degree 4 · about right",
                               "degree 18 · OVERFIT")):
        c = np.polyfit(x, y, deg)
        pred_tr = np.polyval(c, x)
        mse_tr = float(np.mean((pred_tr - y) ** 2))
        xt = rng.uniform(0, 1, 300)
        mse_te = float(np.mean((np.polyval(c, xt) - np.sin(2 * np.pi * xt)) ** 2))
        ax.plot(xs, np.sin(2 * np.pi * xs), color=GREY, lw=1.2, ls="--", label="truth")
        ax.plot(xs, np.polyval(c, xs), color=RED, lw=1.8, label="model")
        ax.scatter(x, y, s=16, color=SLATE, zorder=5)
        ax.set_ylim(-2.1, 2.1)
        ax.set_title(f"{label}\ntrain MSE {mse_tr:.3f} · true MSE {mse_te:.3f}")
        ax.set_xticks([]); ax.set_yticks([])
        rows.append({"degree": deg, "train_mse": round(mse_tr, 4),
                     "test_mse": round(mse_te, 4)})
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    plt.tight_layout()
    return fig, pd.DataFrame(rows)


def main(outdir="figures"):
    out = Path(outdir); out.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    tbl = compare()
    print("Seven models, one dataset. Read the GAP column.\n")
    print(tbl.to_string(index=False), "\n")
    fig_boundaries().savefig(out / "04_boundaries.png", bbox_inches="tight")

    fig, ft = fig_fit()
    fig.savefig(out / "05_fit.png", bbox_inches="tight")
    print("Capacity, on a curve you can see:\n")
    print(ft.to_string(index=False))
    print(f"\nFigures written to {out.resolve()}")


if __name__ == "__main__":
    main()
