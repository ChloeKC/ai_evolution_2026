"""
inductive_bias.py, the thesis of Week 4, measured four ways.

The Evolution of Machine Learning and AI, Week 4 Lab
South East Technological University

The claim being tested:

    An architecture encodes assumptions about its data. When those
    assumptions match reality, the model needs less data and generalises
    better. When they do not, the structure is dead weight or worse.

THE TASK. A 16 by 16 image of faint noise with one small 3 by 3 motif dropped
somewhere in it, at a position chosen at random. Four motifs, so four classes,
and chance is 25%. All four motifs contain exactly five bright pixels, so
total brightness tells you nothing: only the ARRANGEMENT distinguishes them.

    plus        cross       tee         ell
    . X .       X . X       X X X       X . .
    X X X       . X .       . X .       X . .
    . X .       X . X       . X .       X X X

This task is deliberately synthetic, because the point is to isolate one
variable. The signal is local (3 by 3) and it can appear anywhere (196
positions). Those are precisely the two conditions a convolution assumes.

FOUR EXPERIMENTS.

    1  DATA EFFICIENCY. Both models, increasing training sets. Then the MLP
       alone, given far more data and four times the capacity, to find out how
       much data buys what the CNN was given for free.

    2  UNSEEN POSITIONS. Train with the motif only ever in the left third of
       the image. Test with it only in the right third. Three models here, not
       two, because the third one is the uncomfortable case.

    3  THE PIXEL SHUFFLE. One fixed random permutation of the 256 pixels,
       applied to every image, train and test. No information is destroyed:
       the permutation is reversible, and a human given the key could undo it.
       What is destroyed is the MEANING of adjacency.

    4  WHERE IT DOES NOT PAY. The same comparison on small, pre-centred
       digits, where the CNN's advantage disappears. This is not a failure of
       the thesis. It is the thesis.

Booklet cross-references:
    Chapter 2   inductive bias, the concept that runs the week
    Chapter 3   the cost of assuming nothing
    Chapter 7   what a convolution layer assumes
    Chapter 9   pooling, and equivariance against invariance
    Chapter 11  where the CNN's assumptions fail
    Chapter 21  the scaling bargain

Nothing here imports a deep learning library.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys as _sys
import matplotlib
if "ipykernel" not in _sys.modules:      # keep inline plots working in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnn_numpy import make_cnn, make_cnn_flatten, make_mlp

SLATE = "#47546B"
AMBER = "#A96E08"
RED = "#9E3B2C"
GREEN = "#2E7D4F"
BLUE = "#4E7291"
GREY = "#8A93A5"

SIDE = 16
EPOCHS = 120
LR = 0.08
BATCH = 16

MOTIF_NAMES = ("plus", "cross", "tee", "ell")
MOTIFS = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], float),   # plus
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], float),   # cross
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], float),   # tee
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], float),   # ell
]


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------

def make_motifs(n, seed=0, cols=(0, SIDE - 3), rows=(0, SIDE - 3), perm=None):
    """n images, each with one motif at a random position, plus faint noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.12, (n, 1, SIDE, SIDE))
    y = rng.integers(0, len(MOTIFS), n)
    for i in range(n):
        r = rng.integers(rows[0], rows[1] + 1)
        c = rng.integers(cols[0], cols[1] + 1)
        X[i, 0, r:r + 3, c:c + 3] += MOTIFS[y[i]] * rng.uniform(0.85, 1.15)
    X = np.clip(X, -0.5, 1.6)
    if perm is not None:
        X = X.reshape(n, -1)[:, perm].reshape(n, 1, SIDE, SIDE)
    return X, y


def train(builder, Xtr, ytr, seed=0, epochs=EPOCHS, **kw):
    net = builder(side=SIDE, n_classes=len(MOTIFS), seed=seed, **kw)
    net.fit(Xtr, ytr, epochs=epochs, lr=LR, batch=BATCH, seed=seed)
    return net


# ---------------------------------------------------------------------------
# EXPERIMENT 1: data efficiency
# ---------------------------------------------------------------------------

def experiment_data_efficiency(sizes=(64, 128, 256, 512), seeds=(0,),
                               quick=False):
    """Test accuracy against training set size, for both architectures.

    Read the last column. That is the number the week is about, and it is not
    a small effect: it is the difference between a working model and a coin
    flip, on identical data, with the smaller model winning.
    """
    if quick:
        sizes, seeds = (64, 256), (0,)
    Xte, yte = make_motifs(600, seed=99)
    rows = []
    for n in sizes:
        acc = {"CNN": [], "MLP": []}
        tr_acc = {"CNN": [], "MLP": []}
        for s in seeds:
            Xtr, ytr = make_motifs(n, seed=s)
            for name, b in (("CNN", make_cnn), ("MLP", make_mlp)):
                net = train(b, Xtr, ytr, seed=s)
                acc[name].append(net.score(Xte, yte))
                tr_acc[name].append(net.score(Xtr, ytr))
        cnn, mlp = float(np.mean(acc["CNN"])), float(np.mean(acc["MLP"]))
        rows.append({
            "train images": n,
            "CNN train": round(float(np.mean(tr_acc["CNN"])), 3),
            "CNN test": round(cnn, 3),
            "MLP train": round(float(np.mean(tr_acc["MLP"])), 3),
            "MLP test": round(mlp, 3),
            "CNN advantage": f"{(cnn - mlp) * 100:+.1f} pts",
        })
    return pd.DataFrame(rows)


def experiment_mlp_scaling(sizes=(1024, 4096, 16384), hidden=256,
                           quick=False):
    """Give the MLP much more data and four times the capacity. How far does it get?

    This is the fair-hearing experiment. The MLP is not a straw man: it is
    strictly more expressive than the CNN, in the sense that every function
    the CNN can compute is also available to the MLP. It is more powerful and
    it needs orders of magnitude more data, and that is the whole point.

    Note the train column. The MLP reaches 1.000 on its training set almost
    immediately and stays there while its test score crawls. That is Week 3's
    overfitting, arriving with a new cause: not too much capacity in the
    abstract, but capacity pointed in no particular direction.
    """
    if quick:
        sizes = (1024, 16384)
    Xte, yte = make_motifs(600, seed=99)
    rows = []
    for n in sizes:
        Xtr, ytr = make_motifs(n, seed=0)
        net = train(make_mlp, Xtr, ytr, seed=0, epochs=60, hidden=hidden)
        rows.append({
            "MLP train images": n,
            "parameters": net.n_params,
            "train acc": round(net.score(Xtr, ytr), 3),
            "test acc": round(net.score(Xte, yte), 3),
        })
    return pd.DataFrame(rows)


def fig_data_efficiency(eff, scaling, cnn_ref=None):
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    n = eff["train images"]
    ax.plot(n, eff["CNN test"], "o-", color=SLATE, lw=2.6, ms=8,
            label="CNN, 700 parameters (locality and sharing assumed)")
    ax.plot(n, eff["MLP test"], "s-", color=AMBER, lw=2.6, ms=8,
            label="MLP, 16,708 parameters (no spatial assumption)")
    ax.plot(scaling["MLP train images"], scaling["test acc"], "s--", color=AMBER,
            lw=1.8, ms=6, alpha=0.65,
            label="MLP, 66,820 parameters, given far more data")
    ax.axhline(0.25, color=GREY, ls=":", lw=1.4)
    ax.text(70, 0.265, "chance (4 classes)", fontsize=9, color=GREY)

    if cnn_ref is not None:
        ax.axhline(cnn_ref, color=SLATE, ls=":", lw=1.4, alpha=0.8)
        ax.text(2200, cnn_ref + 0.012,
                f"the CNN reached this with {int(n.iloc[1])} images",
                fontsize=9.5, color=SLATE, style="italic")

    ax.set_xscale("log")
    ticks = list(n) + list(scaling["MLP train images"])
    ax.set_xticks(sorted(set(ticks)))
    ax.set_xticklabels([str(v) for v in sorted(set(ticks))], fontsize=9)
    ax.set_xlabel("training images (log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.15, 1.03)
    ax.set_title("What a correct assumption is worth, measured in examples\n"
                 "you did not have to collect",
                 fontsize=12.5, color=SLATE, weight="bold")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# EXPERIMENT 2: positions never seen in training
# ---------------------------------------------------------------------------

def experiment_unseen_positions(n_train=512, seed=0):
    """Train with the motif only on the left. Test with it only on the right.

    Three models, and the middle one is the lesson. `CNN, global pooling` and
    `CNN, flatten head` have IDENTICAL convolutional layers. They differ only
    in the last two lines of the architecture. One transfers to positions it
    has never seen; the other is at chance.

    Weight sharing means the motif detector, once learned on the left, already
    works on the right: the same nine numbers are applied everywhere, so
    "everywhere" comes free. But a flatten-then-dense head reads POSITIONS. It
    learned "this feature, in these slots". Move the feature to slots that
    were always empty during training and the head has nothing to say.

    An assumption is only worth what your architecture does with it all the
    way to the output.
    """
    Xtr, ytr = make_motifs(n_train, seed=seed, cols=(0, 4))
    Xin, yin = make_motifs(400, seed=98, cols=(0, 4))
    Xout, yout = make_motifs(400, seed=97, cols=(9, 13))

    rows = []
    for label, b in (("CNN, global pooling", make_cnn),
                     ("CNN, flatten head", make_cnn_flatten),
                     ("MLP", make_mlp)):
        net = train(b, Xtr, ytr, seed=seed)
        seen, unseen = net.score(Xin, yin), net.score(Xout, yout)
        rows.append({
            "model": label,
            "parameters": net.n_params,
            "positions seen in training": round(seen, 3),
            "positions never seen": round(unseen, 3),
            "drop": round(seen - unseen, 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EXPERIMENT 3: the pixel shuffle
# ---------------------------------------------------------------------------

def experiment_shuffle(n_train=512, seed=0):
    """One fixed permutation of the pixels, applied to every image.

    This is the experiment that settles the argument, because it removes
    exactly one thing and nothing else. Every pixel value survives. Every
    class is still perfectly determined by the image. Only adjacency is gone.

    If the CNN's advantage were general capability, shuffling would cost it
    little. If the advantage was an assumption about neighbours, shuffling
    should erase it completely. Read the table and see which happened.
    """
    rng = np.random.default_rng(7)
    perm = rng.permutation(SIDE * SIDE)

    rows = []
    for label, p in (("original pixel order", None), ("pixels shuffled", perm)):
        entry = {"pixel order": label}
        for name, b in (("CNN", make_cnn), ("MLP", make_mlp)):
            Xtr, ytr = make_motifs(n_train, seed=seed, perm=p)
            Xte, yte = make_motifs(600, seed=99, perm=p)
            net = train(b, Xtr, ytr, seed=seed)
            entry[f"{name} test"] = round(net.score(Xte, yte), 3)
        entry["gap"] = round(entry["CNN test"] - entry["MLP test"], 3)
        rows.append(entry)
    df = pd.DataFrame(rows)
    return df, perm


def fig_shuffle(perm):
    X, y = make_motifs(6, seed=3)
    Xs, _ = make_motifs(6, seed=3, perm=perm)
    fig, axes = plt.subplots(2, 6, figsize=(12.4, 4.6))
    for i in range(6):
        axes[0, i].imshow(X[i, 0], cmap="gray", vmin=-0.5, vmax=1.6)
        axes[0, i].set_title(f"a {MOTIF_NAMES[y[i]]}", fontsize=10)
        axes[1, i].imshow(Xs[i, 0], cmap="gray", vmin=-0.5, vmax=1.6)
        axes[1, i].set_title(f"still a {MOTIF_NAMES[y[i]]}", fontsize=10, color=RED)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Top: the images. Bottom: the same pixel values, one fixed permutation applied.\n"
                 "Nothing was deleted, and the label is still perfectly determined. "
                 "The CNN drops to chance; the MLP does not move.",
                 fontsize=12, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# EXPERIMENT 4: where the assumption does not pay
# ---------------------------------------------------------------------------

def experiment_no_payoff(n_train=640, seed=0):
    """The same two models on small, pre-centred, pre-cropped digits.

    Report this one honestly, because it is the most useful result in the
    file. On 8 by 8 digits that somebody has already centred and cropped for
    you, the CNN's advantage is within noise, and sometimes negative.

    Nothing has gone wrong. Translation equivariance is worth something only
    when there is translation to be equivariant TO. Centring the digits is a
    hand-built solution to the same problem the convolution solves, applied in
    the preprocessing step instead of the architecture, and you do not get
    paid twice for solving a problem once.

    The lesson is not "CNNs are overrated". It is that an architecture's value
    depends on a property of your data, so it is a claim you can check, and
    should. Week 3's field guide asked whether the truth has the shape your
    model can represent. This is that question with a measurement attached.
    """
    from sklearn.datasets import load_digits
    d = load_digits()
    X8 = d.images / 16.0
    y = d.target.astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X8))
    X8, y = X8[idx], y[idx]

    X = X8[:, None, :, :]
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[-500:], y[-500:]

    rows = []
    for label, b in (("CNN, global pooling", make_cnn), ("MLP", make_mlp)):
        net = b(side=8, n_classes=10, seed=seed)
        net.fit(Xtr, ytr, epochs=EPOCHS, lr=LR, batch=BATCH, seed=seed)
        rows.append({
            "model": label,
            "parameters": net.n_params,
            "train acc": round(net.score(Xtr, ytr), 3),
            "test acc": round(net.score(Xte, yte), 3),
        })
    df = pd.DataFrame(rows)
    df["verdict"] = ["the assumption buys little here",
                     "because the data was pre-centred for it"]
    return df


# ---------------------------------------------------------------------------

def fig_summary(eff, unseen, shuf):
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8))

    n = eff["train images"]
    axes[0].plot(n, eff["CNN test"], "o-", color=SLATE, lw=2.4, ms=7, label="CNN (700 params)")
    axes[0].plot(n, eff["MLP test"], "s-", color=AMBER, lw=2.4, ms=7, label="MLP (16,708)")
    axes[0].axhline(0.25, color=GREY, ls=":", lw=1.2)
    axes[0].set_xscale("log")
    axes[0].set_xticks(list(n))
    axes[0].set_xticklabels([str(v) for v in n], fontsize=9)
    axes[0].set_xlabel("training images")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_ylim(0.15, 1.03)
    axes[0].set_title("1. Scarce data\nthe right assumption pays", fontsize=11, color=SLATE)
    axes[0].legend(fontsize=8.5, loc="center right")
    axes[0].grid(alpha=0.25)

    x = np.arange(len(unseen))
    axes[1].bar(x - 0.19, unseen["positions seen in training"], 0.38,
                color=SLATE, label="positions seen")
    axes[1].bar(x + 0.19, unseen["positions never seen"], 0.38,
                color=AMBER, label="positions never seen")
    axes[1].axhline(0.25, color=GREY, ls=":", lw=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["CNN\nglobal pool", "CNN\nflatten head", "MLP"], fontsize=9)
    axes[1].set_ylabel("test accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("2. Positions never seen\nthe head can throw it away",
                      fontsize=11, color=SLATE)
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.25, axis="y")

    x = np.arange(2)
    axes[2].bar(x - 0.19, shuf["CNN test"], 0.38, color=SLATE, label="CNN")
    axes[2].bar(x + 0.19, shuf["MLP test"], 0.38, color=AMBER, label="MLP")
    axes[2].axhline(0.25, color=GREY, ls=":", lw=1.2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["original\npixel order", "pixels\nshuffled"], fontsize=9)
    axes[2].set_ylabel("test accuracy")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("3. Adjacency destroyed\nthe assumption WAS the advantage",
                      fontsize=11, color=RED)
    axes[2].legend(fontsize=8.5)
    axes[2].grid(alpha=0.25, axis="y")

    fig.suptitle("Structure is an assumption about the world. Three measurements of what that means.",
                 fontsize=13, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


def main(quick=False):
    print("\nEXPERIMENT 1: data efficiency")
    eff = experiment_data_efficiency(quick=quick)
    print(eff.to_string(index=False))

    print("\n  ...and the MLP, given much more data and four times the capacity")
    scaling = experiment_mlp_scaling(quick=quick)
    print(scaling.to_string(index=False))

    best_mlp = scaling["test acc"].max()
    cnn_small = eff.loc[1, "CNN test"]
    n_small = int(eff.loc[1, "train images"])
    n_big = int(scaling.loc[scaling["test acc"].idxmax(), "MLP train images"])
    print(f"\n  The CNN scored {cnn_small:.3f} from {n_small} images.")
    print(f"  The MLP, with {n_big} images and 66,820 parameters, scored {best_mlp:.3f}.")
    print(f"  That is a factor of about {n_big // n_small} in data, and the MLP is still behind.")

    print("\nEXPERIMENT 2: positions never seen in training")
    unseen = experiment_unseen_positions()
    print(unseen.to_string(index=False))

    print("\nEXPERIMENT 3: one fixed pixel permutation")
    shuf, perm = experiment_shuffle()
    print(shuf.to_string(index=False))
    print("\n  The CNN's advantage was not general capability. It was an assumption")
    print("  about neighbouring pixels, and the shuffle removed exactly that.")

    print("\nEXPERIMENT 4: where the assumption does NOT pay")
    nopay = experiment_no_payoff()
    print(nopay.to_string(index=False))

    return eff, scaling, unseen, shuf, perm, nopay


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
