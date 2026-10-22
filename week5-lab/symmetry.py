"""
symmetry.py, the strongest prior in the week, because it is exactly true.

The Evolution of Machine Learning and AI, Week 5 Lab
South East Technological University

WHY SYMMETRY IS DIFFERENT FROM EVERY OTHER PRIOR IN THIS MODULE

Week 4's convolution assumed that meaning is local. That is approximately true of
photographs, and the approximation is good enough to be worth 128 times the data.

A symmetry is not approximately true. The energy of a molecule does not change
when you rotate it. Not "changes only slightly": does not change, at all, ever,
as a matter of physics rather than of statistics. There is no dataset, no
distribution shift and no adversarial example that can make it false.

So this is the cleanest case of the week's thesis. When you can build in a
constraint that is exactly true, you get an exact guarantee out, and the model
cannot violate it even when the data is noisy, sparse or unrepresentative.

THE TASK

Three particles in a plane. Their potential energy is a sum over pairs of a Morse
potential, which depends only on the DISTANCE between each pair:

    V(r) = exp(-2(r - 1)) - 2 exp(-(r - 1))

The input is six numbers, the raw coordinates. The target is the total energy.

Because energy depends only on distances, it is unchanged by:

    translation     move all three particles together
    rotation        turn the whole arrangement
    permutation     relabel which particle is which

THE TWO MODELS

    COORDINATE MLP    reads the six raw coordinates. It must LEARN that rotating
                      the input does not change the answer, from examples.

    INVARIANT MODEL   computes the three pairwise distances first, sorts them,
                      and feeds those to the same size of network. Rotating the
                      input cannot change a distance, so the output cannot change.
                      The invariance is arithmetic, not learned.

The second model is smaller in input width and otherwise identical. Everything
below measures what that one design decision buys.

Booklet cross-references: Chapters 14 and 15.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys as _sys
import matplotlib
if "ipykernel" not in _sys.modules:      # keep inline plots working in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autodiff import Node, grad, sub, square, mean, const
from mlp import MLP, Adam

SLATE = "#47546B"
AMBER = "#A96E08"
RED = "#9E3B2C"
GREEN = "#2E7D4F"
GREY = "#8A93A5"

N_PARTICLES = 3


# ---------------------------------------------------------------------------
# The physics
# ---------------------------------------------------------------------------

def morse(r):
    """A smooth, bounded pair potential with a minimum at r = 1."""
    return np.exp(-2.0 * (r - 1.0)) - 2.0 * np.exp(-(r - 1.0))


def energy(coords):
    """Total potential energy. coords has shape (N, n_particles, 2)."""
    coords = np.asarray(coords, float)
    n = coords.shape[1]
    total = np.zeros((coords.shape[0], 1))
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(coords[:, i] - coords[:, j], axis=1, keepdims=True)
            total += morse(np.clip(r, 0.35, None))
    return total


def pair_distances(coords):
    """The invariant features: every pairwise distance, sorted.

    Sorting is what buys PERMUTATION invariance on top of rotation and
    translation. Without it, relabelling the particles would reorder the feature
    vector and the network would see a different input for the same physical
    arrangement.
    """
    coords = np.asarray(coords, float)
    n = coords.shape[1]
    ds = []
    for i in range(n):
        for j in range(i + 1, n):
            ds.append(np.linalg.norm(coords[:, i] - coords[:, j], axis=1))
    d = np.stack(ds, axis=1)
    return np.sort(d, axis=1)


def make_data(n, seed=0, spread=0.35, rotate=False, rng=None):
    """Random arrangements of three particles, and their energies.

    NOTE ON A DESIGN TRAP WORTH KNOWING ABOUT. The obvious way to write this is
    to draw all six coordinates from an isotropic Gaussian and then rotate by a
    restricted angle to simulate a dataset that only saw some orientations. That
    does not work, and the first version of this file made exactly that mistake.

    An isotropic Gaussian is ALREADY rotationally symmetric, so restricting the
    rotation angle changes nothing at all: the distribution of arrangements is
    identical either way, and the "unseen orientations" experiment measured
    nothing. The bug was invisible because the code looked right.

    So the arrangement is built with a definite orientation instead: three
    particles at fixed angular positions with random radii and small jitter, which
    gives a shape that genuinely points somewhere. THEN the global rotation is
    applied, and restricting it now restricts something real.

    The lesson generalises past this file. If you want to test whether a model has
    learned an invariance, your data has to be capable of violating it.
    """
    rng = rng or np.random.default_rng(seed)
    base_angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    radii = rng.uniform(0.7, 1.6, (n, N_PARTICLES))
    jitter = rng.normal(0, spread, (n, N_PARTICLES))
    ang = base_angles[None, :] + jitter
    coords = np.stack([radii * np.cos(ang), radii * np.sin(ang)], axis=-1)
    if rotate:
        theta = rng.uniform(0, 2 * np.pi, n)
    else:
        theta = rng.uniform(0, np.pi / 6, n)      # a narrow band of orientations
    c, s = np.cos(theta), np.sin(theta)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    coords = np.einsum("nij,npj->npi", R, coords)
    return coords, energy(coords)


def rotate_batch(coords, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return np.einsum("ij,npj->npi", R, coords)


# ---------------------------------------------------------------------------
# The two models
# ---------------------------------------------------------------------------

class CoordinateNet:
    """Reads raw coordinates. Must learn the symmetry from examples."""

    name = "coordinate MLP"

    def __init__(self, hidden=48, seed=0):
        self.net = MLP([N_PARTICLES * 2, hidden, hidden, 1], seed=seed)
        self.params = self.net.params

    def features(self, coords):
        return np.asarray(coords, float).reshape(len(coords), -1)

    def __call__(self, coords):
        return self.net(Node(self.features(coords)))

    @property
    def n_params(self):
        return self.net.n_params


class InvariantNet:
    """Reads sorted pairwise distances. The symmetry holds by construction.

    This is the same trick as Week 4's convolution, applied to a different group.
    A convolution shares weights across translations in space, so translating the
    input translates the output. Here the features themselves are unchanged by the
    symmetry, so transforming the input does not change the output at all: this is
    INVARIANCE rather than equivariance, which is the distinction Week 4,
    Chapter 9 laboured over.
    """

    name = "invariant (distances)"

    def __init__(self, hidden=48, seed=0):
        n_pairs = N_PARTICLES * (N_PARTICLES - 1) // 2
        self.net = MLP([n_pairs, hidden, hidden, 1], seed=seed)
        self.params = self.net.params

    def features(self, coords):
        return pair_distances(coords)

    def __call__(self, coords):
        return self.net(Node(self.features(coords)))

    @property
    def n_params(self):
        return self.net.n_params


def fit_energy(model, coords, y, epochs=1500, lr=3e-3, batch=64, seed=0):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params, lr=lr)
    X = model.features(coords)
    for ep in range(epochs):
        idx = rng.permutation(len(X))[:batch]
        pred = model.net(Node(X[idx]))
        loss = mean(square(sub(pred, const(y[idx]))))
        opt.step(grad(loss, model.params))
        if ep + 1 in (int(epochs * 0.6), int(epochs * 0.85)):
            opt.set_lr(opt.lr * 0.3)
    return model


def rmse(model, coords, y):
    pred = model(coords).value
    return float(np.sqrt(np.mean((pred - y) ** 2)))


# ---------------------------------------------------------------------------
# EXPERIMENT 1: exact invariance, tested rather than asserted
# ---------------------------------------------------------------------------

def experiment_invariance(seed=0):
    """Rotate the input by 90 degrees and see whether the prediction changes.

    Week 4 ran the equivalent test on convolution and got exactly 0.00e+00. The
    same thing happens here, and for the same reason: the quantity the network
    reads is unchanged by the transformation, so no training is involved in the
    guarantee.
    """
    coords, y = make_data(400, seed=seed, rotate=True)
    rows = []
    for ctor in (CoordinateNet, InvariantNet):
        m = ctor(seed=seed)
        fit_energy(m, *make_data(2000, seed=seed + 1, rotate=True), epochs=1200)
        base = m(coords).value
        worst = 0.0
        for theta in (np.pi / 2, np.pi, 0.7, 2.3):
            rot = m(rotate_batch(coords, theta)).value
            worst = max(worst, float(np.abs(rot - base).max()))
        scale = float(np.abs(base).max())
        rows.append({"model": m.name, "parameters": m.n_params,
                     "max change under rotation": f"{worst:.2e}",
                     "as % of output scale": f"{100 * worst / scale:.1f}%",
                     "verdict": ("invariant by construction" if worst < 1e-12
                                 else "invariance only approximate")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EXPERIMENT 2: data efficiency
# ---------------------------------------------------------------------------

def experiment_symmetry_efficiency(sizes=(64, 256, 1024, 4096), epochs=1500,
                                   seed=0, quick=False):
    """How much data does each model need to learn the energy?

    Both models see the same arrangements. One of them already knows that
    orientation is irrelevant; the other has to work it out, and working it out
    costs examples that could have been spent learning the actual physics.
    """
    if quick:
        sizes, epochs = (64, 1024), 900
    test_coords, test_y = make_data(1500, seed=999, rotate=True)
    rows = []
    for n in sizes:
        tr = make_data(n, seed=seed, rotate=True)
        row = {"training arrangements": n}
        for ctor in (CoordinateNet, InvariantNet):
            m = ctor(seed=seed)
            fit_energy(m, *tr, epochs=epochs, seed=seed)
            row[ctor.name + " RMSE"] = round(rmse(m, test_coords, test_y), 4)
        row["ratio"] = round(row["coordinate MLP RMSE"] /
                             max(row["invariant (distances) RMSE"], 1e-12), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EXPERIMENT 3: orientations never seen in training
# ---------------------------------------------------------------------------

def experiment_unseen_orientations(n_train=2048, epochs=1500, seed=0,
                                   quick=False):
    """Train on a narrow band of orientations. Test on all of them.

    This is the realistic version of the problem, and it is Week 4's
    "positions never seen in training" experiment with rotation substituted for
    translation. A dataset that happens to contain molecules in one range of
    orientations is not unusual; it is normal.
    """
    if quick:
        epochs = 900
    tr = make_data(n_train, seed=seed, rotate=False)      # narrow band
    seen = make_data(800, seed=seed + 7, rotate=False)    # same band
    unseen = make_data(800, seed=seed + 8, rotate=True)   # all orientations
    rows = []
    for ctor in (CoordinateNet, InvariantNet):
        m = ctor(seed=seed)
        fit_energy(m, *tr, epochs=epochs, seed=seed)
        a, b = rmse(m, *seen), rmse(m, *unseen)
        rows.append({"model": m.name,
                     "RMSE, orientations seen": round(a, 4),
                     "RMSE, all orientations": round(b, 4),
                     "degradation factor": round(b / max(a, 1e-12), 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EXPERIMENT 4: the honest cost of a symmetry you do not actually have
# ---------------------------------------------------------------------------

def experiment_wrong_symmetry(n_train=2048, epochs=1500, seed=0, quick=False):
    """What if the target is NOT invariant, and you assumed it was?

    Add a term to the energy that depends on absolute height, as gravity would.
    Now the true energy DOES change when you rotate the arrangement, so the
    invariant model is structurally unable to represent the truth: it has thrown
    away the information it needs, and no amount of data can recover it.

    This is the counterpart of pinn.py's wrong-damping experiment, and it makes
    the same point in a form that cannot be fixed by training longer. An
    architectural constraint is a promise about the world, and if the promise is
    false the model is not merely worse, it is incapable.
    """
    if quick:
        epochs = 900

    def gravity_energy(coords):
        return energy(coords) + 0.4 * coords[:, :, 1].sum(axis=1, keepdims=True)

    tr_c, _ = make_data(n_train, seed=seed, rotate=True)
    te_c, _ = make_data(800, seed=seed + 33, rotate=True)
    tr_y, te_y = gravity_energy(tr_c), gravity_energy(te_c)

    rows = []
    for ctor in (CoordinateNet, InvariantNet):
        m = ctor(seed=seed)
        fit_energy(m, tr_c, tr_y, epochs=epochs, seed=seed)
        rows.append({"model": m.name,
                     "RMSE on a target that is NOT invariant": round(rmse(m, te_c, te_y), 4),
                     "verdict": ("can represent the height term"
                                 if ctor is CoordinateNet
                                 else "structurally cannot, at any dataset size")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_symmetry_efficiency(df):
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.loglog(df["training arrangements"], df["coordinate MLP RMSE"], "o-",
              lw=2.4, color=AMBER, label="coordinate MLP (must learn the symmetry)")
    ax.loglog(df["training arrangements"], df["invariant (distances) RMSE"], "s-",
              lw=2.4, color=SLATE, label="invariant model (symmetry built in)")
    ax.set_xlabel("training arrangements")
    ax.set_ylabel("RMSE on unseen arrangements")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=10)
    ax.set_title("The same network, the same data, one fact supplied in advance",
                 fontsize=12.5, color=SLATE)
    fig.tight_layout()
    return fig


def fig_rotation_sweep(seed=0):
    """One arrangement, rotated through a full turn. Does the prediction move?"""
    coords, y = make_data(1, seed=42, rotate=True)
    thetas = np.linspace(0, 2 * np.pi, 120)
    preds = {}
    for ctor in (CoordinateNet, InvariantNet):
        m = ctor(seed=seed)
        fit_energy(m, *make_data(2048, seed=seed, rotate=False), epochs=1200)
        preds[m.name] = np.array(
            [m(rotate_batch(coords, th)).value.item() for th in thetas])
    truth = energy(coords).item()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(truth, color=GREY, lw=3.2, label=f"the truth: constant at {truth:.3f}")
    ax.plot(thetas, preds["coordinate MLP"], lw=2.2, color=AMBER,
            label="coordinate MLP")
    ax.plot(thetas, preds["invariant (distances)"], lw=2.2, color=SLATE, ls="--",
            label="invariant model")
    ax.axvspan(0, np.pi / 6, color=GREEN, alpha=0.10)
    ax.text(np.pi / 12, ax.get_ylim()[1] * 0.96, "orientations\nseen in training",
            ha="center", va="top", fontsize=9.5, color=GREEN)
    ax.set_xlabel("rotation applied to the arrangement (radians)")
    ax.set_ylabel("predicted energy")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_title("Rotating a molecule cannot change its energy.\n"
                 "One of these models knows that as arithmetic; the other guessed from data.",
                 fontsize=12.5, color=SLATE)
    fig.tight_layout()
    return fig


def main(quick=False):
    print("\nEXPERIMENT 1: is the invariance exact?")
    inv = experiment_invariance()
    print(inv.to_string(index=False))

    print("\nEXPERIMENT 2: data efficiency")
    eff = experiment_symmetry_efficiency(quick=quick)
    print(eff.to_string(index=False))

    print("\nEXPERIMENT 3: orientations never seen in training")
    un = experiment_unseen_orientations(quick=quick)
    print(un.to_string(index=False))

    print("\nEXPERIMENT 4: when the symmetry does not hold")
    wr = experiment_wrong_symmetry(quick=quick)
    print(wr.to_string(index=False))
    return inv, eff, un, wr


if __name__ == "__main__":
    import sys as _s
    main(quick="--quick" in _s.argv)
