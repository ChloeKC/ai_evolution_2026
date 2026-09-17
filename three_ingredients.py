"""
three_ingredients.py — the Week 1 thesis, as an experiment you can run.

The Evolution of Machine Learning and AI · Week 1 Lab
South East Technological University

The lecture claims that deep learning ignited only when DATA, COMPUTE and
ARCHITECTURE coexisted, and that every partial combination has a name and an
era attached. That is a historical claim. This script turns it into a
measurable one, on a dataset small enough to run on a laptop in a minute.

    data          = number of training examples
    architecture  = model capacity (hidden units)
    compute       = training budget (optimiser iterations)

Switch each to LOW or HIGH and read the accuracy. You should find:

    architecture only   the perceptron, 1958: right idea, starved
    compute only        Deep Blue, 1997: power, nothing to learn from
    data + compute      ImageNet 2010: an arena with no champion
    all three           AlexNet, 2012

No GPU, no download, no network: sklearn ships the digits dataset.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SETTINGS = {
    "data":         {"low": 120,       "high": 1400},
    "architecture": {"low": (2,),      "high": (128, 64)},
    "compute":      {"low": 8,         "high": 600},
}

ERA_NAMES = {
    (False, False, False): "nothing at all",
    (True,  False, False): "data alone: the 2000s web, shallow ceiling",
    (False, True,  False): "architecture alone: the perceptron, 1958, starved",
    (False, False, True):  "compute alone: Deep Blue, 1997, no transfer",
    (True,  True,  False): "data + architecture, no compute: 1990s neural nets, too slow",
    (True,  False, True):  "data + compute, no architecture: ImageNet 2010, an arena with no champion",
    (False, True,  True):  "architecture + compute, no data: a very fast way to overfit",
    (True,  True,  True):  "all three: AlexNet, 2012",
}


def run_one(data_hi: bool, arch_hi: bool, compute_hi: bool, seed: int = 0) -> dict:
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)

    n = SETTINGS["data"]["high" if data_hi else "low"]
    n = min(n, len(Xtr))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=n, replace=False)
    Xtr, ytr = Xtr[idx], ytr[idx]

    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(
        hidden_layer_sizes=SETTINGS["architecture"]["high" if arch_hi else "low"],
        max_iter=SETTINGS["compute"]["high" if compute_hi else "low"],
        random_state=seed, learning_rate_init=0.01,
    )
    clf.fit(sc.transform(Xtr), ytr)

    return {
        "data": "HIGH" if data_hi else "low",
        "architecture": "HIGH" if arch_hi else "low",
        "compute": "HIGH" if compute_hi else "low",
        "n_train": n,
        "hidden": str(SETTINGS["architecture"]["high" if arch_hi else "low"]),
        "iters": SETTINGS["compute"]["high" if compute_hi else "low"],
        "train_acc": float(clf.score(sc.transform(Xtr), ytr)),
        "test_acc": float(clf.score(sc.transform(Xte), yte)),
        "era": ERA_NAMES[(data_hi, arch_hi, compute_hi)],
    }


def run_all(seeds=(0, 1, 2)) -> pd.DataFrame:
    """All eight combinations, averaged over seeds."""
    rows = []
    for d in (False, True):
        for a in (False, True):
            for c in (False, True):
                runs = [run_one(d, a, c, seed=s) for s in seeds]
                base = runs[0].copy()
                base["train_acc"] = float(np.mean([r["train_acc"] for r in runs]))
                base["test_acc"] = float(np.mean([r["test_acc"] for r in runs]))
                base["test_sd"] = float(np.std([r["test_acc"] for r in runs]))
                rows.append(base)
    df = pd.DataFrame(rows).sort_values("test_acc").reset_index(drop=True)
    return df


def fig_ingredients_grid(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SLATE, AMBER, GREY = "#47546B", "#A96E08", "#8A93A5"
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    labels = [f"{r.data[0]}{r.architecture[0]}{r.compute[0]}  {r.era[:44]}"
              for _, r in df.iterrows()]
    colours = [AMBER if r.era.startswith("all three") else SLATE for _, r in df.iterrows()]
    ax.barh(range(len(df)), df.test_acc, color=colours)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=7.6)
    for i, v in enumerate(df.test_acc):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=7.6, color=GREY)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("held-out accuracy")
    ax.set_title("Every partial combination has a name, and each one stalls\n"
                 "(D/A/C = data / architecture / compute; capital = HIGH)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    from pathlib import Path

    df = run_all()
    cols = ["data", "architecture", "compute", "n_train", "hidden", "iters",
            "train_acc", "test_acc", "era"]
    pd.set_option("display.width", 170)
    print(df[cols].round(3).to_string(index=False))

    out = Path("figures"); out.mkdir(exist_ok=True)
    fig_ingredients_grid(df).savefig(out / "05_three_ingredients.png", bbox_inches="tight")

    best = df.iloc[-1]
    print(f"\nBest: {best.era}  ({best.test_acc:.3f})")
    print("""
Two things to notice, and they are both in the lecture:

  * The high-capacity, high-compute, low-data run has excellent TRAINING
    accuracy and poor test accuracy. That is a very fast way to memorise,
    and it is what the 1990s could afford.
  * No single ingredient carries the result. The jump comes from the
    combination, which is exactly the claim the 2012 slide makes:
    'a coincidence of supply; almost no new ideas'.""")
