"""
diffusion.py, a generative model that is a thermodynamics paper wearing a hat.

The Evolution of Machine Learning and AI, Week 5 Lab
South East Technological University

WHERE THIS CAME FROM

Every image generator you have used descends from a 2015 paper by Sohl-Dickstein
and colleagues titled "Deep Unsupervised Learning using Nonequilibrium
Thermodynamics". The title is not a metaphor and not marketing. The method is
taken directly from statistical physics, and understanding it that way makes the
algorithm obvious rather than mysterious.

THE PHYSICAL PICTURE

Put a drop of ink in a glass of water. Molecular collisions push the ink about at
random, the sharp structure of the drop dissolves, and after long enough the glass
holds a featureless uniform solution. That is diffusion driving a system to
EQUILIBRIUM, and it destroys information: from the final glass you cannot tell
where the drop was.

The forward process here is exactly that, applied to a dataset. Add a little
Gaussian noise, repeatedly. Structure dissolves. After enough steps the data is
indistinguishable from an isotropic Gaussian, which is the equilibrium
distribution of this particular process.

Now the trick. Each individual step added only a LITTLE noise, so each individual
step is nearly reversible: given a slightly noisy sample, you can estimate what
was removed. Learn that one small reversal, apply it many times, and you run the
diffusion backwards. Start from pure noise, denoise repeatedly, and you arrive at
something that looks like the data.

    forward       data  ->  noise      known, fixed, no learning at all
    reverse       noise ->  data       learned, one small step at a time

WHY THIS BELONGS IN WEEK 5

The forward process is not learned. It is HANDED to the model, from physics. That
is a structural prior in exactly the sense of this whole week: a piece of the
problem that nobody had to fit, which makes the remaining learning much easier.

Compare it with the alternative. A generative adversarial network learns the whole
map from noise to data in one leap, with nothing given. It works, and it is
notoriously unstable to train. A diffusion model breaks the same journey into
hundreds of tiny steps whose forward direction is known exactly, and each step is
an ordinary regression. Stability came from structure, not from a better optimiser.

THE MATHEMATICS THIS CONNECTS TO

The reverse of a diffusion needs the SCORE, the gradient of the log density. That
is Langevin dynamics, which physicists have used to sample from Boltzmann
distributions since the 1900s. The forward process obeys the Fokker-Planck
equation. None of this vocabulary was invented for machine learning.

Booklet cross-references: Chapters 20 and 21.
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


# ---------------------------------------------------------------------------
# Data: a shape that a Gaussian cannot possibly imitate
# ---------------------------------------------------------------------------

def two_moons(n, noise=0.06, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n - n1)
    a = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    b = np.stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5], axis=1)
    x = np.concatenate([a, b], axis=0) + rng.normal(0, noise, (n, 2))
    x = (x - x.mean(0)) / x.std(0)
    return x


# ---------------------------------------------------------------------------
# The forward process, which is given rather than learned
# ---------------------------------------------------------------------------

class Schedule:
    """The noise schedule. Fixed in advance; nothing here is trained.

    beta_t is how much variance is injected at step t. alpha_bar_t is how much of
    the ORIGINAL signal survives to step t, and it is what makes the whole method
    practical: instead of simulating t steps one at a time, you can jump straight
    to step t in closed form,

        x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps

    which is why training is cheap even with a thousand steps.
    """

    def __init__(self, n_steps=200, beta_start=1e-4, beta_end=0.06):
        self.n_steps = n_steps
        self.beta = np.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)

    def noise(self, x0, t_idx, rng):
        ab = self.alpha_bar[t_idx][:, None]
        eps = rng.normal(0, 1, x0.shape)
        return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps, eps


class Denoiser:
    """Predicts the noise that was added. That is the only learned component.

    Why predict the noise rather than the clean sample? They are equivalent by
    algebra, but the noise target has constant unit variance at every step, so one
    network with one learning rate handles every noise level. Predicting the clean
    sample gives a target whose scale changes by orders of magnitude across the
    schedule. A small reparameterisation that makes the optimisation behave, in
    the same family as Week 3's feature scaling.
    """

    def __init__(self, hidden=64, n_steps=200, seed=0):
        self.net = MLP([3, hidden, hidden, hidden, 2], seed=seed)
        self.params = self.net.params
        self.n_steps = n_steps

    @property
    def n_params(self):
        return self.net.n_params

    def features(self, x, t_idx):
        tt = (np.asarray(t_idx, float).reshape(-1, 1) / self.n_steps) * 2.0 - 1.0
        return np.concatenate([np.asarray(x, float), tt], axis=1)

    def __call__(self, x, t_idx):
        return self.net(Node(self.features(x, t_idx)))

    def predict(self, x, t_idx):
        return self(x, t_idx).value


def train_diffusion(model, data, sched, epochs=3000, lr=3e-3, batch=256, seed=0):
    """One line of physics, one line of regression.

    Pick a random step, jump the data to that noise level in closed form, and ask
    the network to name the noise. That is the entire training objective, and it
    is a plain squared error. No adversary, no discriminator, no instability.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(model.params, lr=lr)
    hist = []
    for ep in range(epochs):
        idx = rng.integers(0, len(data), batch)
        t_idx = rng.integers(0, sched.n_steps, batch)
        x_t, eps = sched.noise(data[idx], t_idx, rng)
        pred = model(x_t, t_idx)
        loss = mean(square(sub(pred, const(eps))))
        opt.step(grad(loss, model.params))
        hist.append(loss.value.item())
        if ep + 1 in (int(epochs * 0.6), int(epochs * 0.85)):
            opt.set_lr(opt.lr * 0.3)
    return hist


@np.errstate(all="ignore")
def sample(model, sched, n=800, seed=0, keep=None):
    """Run the diffusion backwards, starting from equilibrium.

    This is ancestral sampling: at each step, estimate the noise, remove the right
    fraction of it, and add back a little fresh noise. The last part surprises
    people, but it is the same reason a physical system explores its state space
    rather than collapsing to one point.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2))
    keep = keep or []
    snapshots = {}
    for t in reversed(range(sched.n_steps)):
        eps = model.predict(x, np.full(n, t))
        a, ab, b = sched.alpha[t], sched.alpha_bar[t], sched.beta[t]
        mean_x = (x - b / np.sqrt(1 - ab) * eps) / np.sqrt(a)
        x = mean_x + (np.sqrt(b) * rng.normal(0, 1, x.shape) if t > 0 else 0.0)
        if t in keep:
            snapshots[t] = x.copy()
    return x, snapshots


# ---------------------------------------------------------------------------
# Measuring how good the samples are
# ---------------------------------------------------------------------------

def energy_distance(a, b, seed=0, m=800):
    """A proper distance between two sets of samples, and easy to trust.

        E = 2 mean|a - b|  -  mean|a - a'|  -  mean|b - b'|

    It is zero if and only if the two distributions are identical, needs no
    density estimate and no bandwidth to tune. Lower is better.
    """
    rng = np.random.default_rng(seed)
    a = a[rng.permutation(len(a))[:m]]
    b = b[rng.permutation(len(b))[:m]]

    def md(u, v):
        return np.mean(np.linalg.norm(u[:, None, :] - v[None, :, :], axis=-1))

    return float(2 * md(a, b) - md(a, a) - md(b, b))


def experiment_diffusion(step_counts=(10, 50, 200), epochs=3000, seed=0,
                         quick=False):
    """Sample quality against the number of reverse steps, plus two baselines.

    The Gaussian baseline is the point of comparison worth understanding. Fitting
    a Gaussian to two crescents gives you the right mean and the right covariance
    and a completely wrong shape. The diffusion model has no more information than
    the Gaussian did; it simply had the forward process handed to it, and that was
    enough to turn an intractable generative problem into a sequence of ordinary
    regressions.
    """
    if quick:
        step_counts, epochs = (10, 200), 1200
    data = two_moons(3000, seed=seed)
    held = two_moons(3000, seed=seed + 1)
    rows = []

    rng = np.random.default_rng(seed)
    gauss = rng.multivariate_normal(data.mean(0), np.cov(data.T), 800)
    rows.append({"model": "a fitted Gaussian", "reverse steps": "n/a",
                 "energy distance to the truth": round(energy_distance(gauss, held), 4)})
    rows.append({"model": "pure noise (the equilibrium)", "reverse steps": "n/a",
                 "energy distance to the truth":
                     round(energy_distance(rng.normal(0, 1, (800, 2)), held), 4)})

    models = {}
    for ns in step_counts:
        sched = Schedule(n_steps=ns)
        m = Denoiser(n_steps=ns, seed=seed)
        train_diffusion(m, data, sched, epochs=epochs, seed=seed)
        gen, _ = sample(m, sched, n=800, seed=seed)
        models[ns] = (m, sched)
        rows.append({"model": "diffusion model", "reverse steps": ns,
                     "energy distance to the truth":
                         round(energy_distance(gen, held), 4)})

    floor = energy_distance(data, held)
    rows.append({"model": "two real samples of the data (the floor)",
                 "reverse steps": "n/a",
                 "energy distance to the truth": round(floor, 4)})
    df = pd.DataFrame(rows)
    df["times the floor"] = (df["energy distance to the truth"] / floor).round(2)
    return df, models, data


def experiment_equilibrium(seed=0):
    """Watch the forward process destroy the data and reach equilibrium.

    Nothing is learned here. This is the physics half of the method, and it is
    worth seeing that it really does converge to an isotropic Gaussian, because
    that is the distribution the reverse process starts from.
    """
    data = two_moons(2000, seed=seed)
    sched = Schedule(n_steps=200)
    rng = np.random.default_rng(seed)
    target = rng.normal(0, 1, (2000, 2))
    rows = []
    for t in (0, 10, 40, 100, 199):
        x, _ = sched.noise(data, np.full(len(data), t), rng)
        rows.append({"forward step": t,
                     "fraction of signal left (sqrt of alpha-bar)":
                         round(float(np.sqrt(sched.alpha_bar[t])), 4),
                     "energy distance to a standard Gaussian":
                         round(energy_distance(x, target, seed=seed), 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_forward(seed=0):
    data = two_moons(1200, seed=seed)
    sched = Schedule(n_steps=200)
    rng = np.random.default_rng(seed)
    steps = (0, 10, 40, 100, 199)
    fig, axes = plt.subplots(1, len(steps), figsize=(14, 3.1))
    for ax, t in zip(axes, steps):
        x, _ = sched.noise(data, np.full(len(data), t), rng)
        ax.scatter(x[:, 0], x[:, 1], s=3.5, color=SLATE, alpha=0.55)
        ax.set_title(f"step {t}", fontsize=10.5, color=SLATE)
        ax.set_xlim(-3.4, 3.4)
        ax.set_ylim(-3.4, 3.4)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("The forward process: structure dissolving into equilibrium. "
                 "Nothing here is learned.",
                 fontsize=12.5, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


def fig_reverse(model, sched, data, seed=0):
    keep = [int(sched.n_steps * f) for f in (0.95, 0.6, 0.3, 0.1)] + [0]
    gen, snaps = sample(model, sched, n=1200, seed=seed, keep=keep)
    order = sorted(snaps.keys(), reverse=True)
    fig, axes = plt.subplots(1, len(order) + 1, figsize=(14.5, 3.1))
    for ax, t in zip(axes, order):
        x = snaps[t]
        ax.scatter(x[:, 0], x[:, 1], s=3.5, color=AMBER, alpha=0.55)
        ax.set_title(f"step {t} of {sched.n_steps}", fontsize=10, color=SLATE)
        ax.set_xlim(-3.4, 3.4)
        ax.set_ylim(-3.4, 3.4)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1].scatter(data[:1200, 0], data[:1200, 1], s=3.5, color=GREY, alpha=0.55)
    axes[-1].set_title("the real data", fontsize=10, color=SLATE)
    axes[-1].set_xlim(-3.4, 3.4)
    axes[-1].set_ylim(-3.4, 3.4)
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])
    fig.suptitle("The reverse process: the same diffusion, run backwards by a learned denoiser",
                 fontsize=12.5, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


def fig_score(model, sched, seed=0):
    """The learned denoising direction is, up to scaling, the score of the data."""
    g = np.linspace(-2.6, 2.6, 19)
    X, Y = np.meshgrid(g, g)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    t = int(sched.n_steps * 0.25)
    eps = model.predict(pts, np.full(len(pts), t))
    direction = -eps / (np.linalg.norm(eps, axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.quiver(X, Y, direction[:, 0].reshape(X.shape), direction[:, 1].reshape(X.shape),
              color=SLATE, alpha=0.8, width=0.004)
    d = two_moons(900, seed=seed)
    ax.scatter(d[:, 0], d[:, 1], s=5, color=AMBER, alpha=0.5)
    ax.set_title("Which way is 'towards the data'?\n"
                 "The arrows are the learned score, which is Langevin dynamics' ingredient.",
                 fontsize=11.5, color=SLATE)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def main(quick=False):
    print("\nThe forward process reaching equilibrium (nothing learned here):")
    eq = experiment_equilibrium()
    print(eq.to_string(index=False))
    print("\nSample quality against the number of reverse steps:")
    df, models, data = experiment_diffusion(quick=quick)
    print(df.to_string(index=False))
    print("\n  Read the last row first: two independent samples of the real data")
    print("  differ by that much, so it is the floor. Nothing can score below it.")
    return eq, df, models, data


if __name__ == "__main__":
    import sys as _s
    main(quick="--quick" in _s.argv)
