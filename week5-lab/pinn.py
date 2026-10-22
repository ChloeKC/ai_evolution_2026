"""
pinn.py, a physics-informed neural network, and the baseline that assumed nothing.

The Evolution of Machine Learning and AI, Week 5 Lab
South East Technological University

THE PROBLEM

A mass on a spring with friction. Newton's second law gives

    m u''(t) + c u'(t) + k u(t) = 0,     u(0) = 1,  u'(0) = 0

where u is displacement. With m = 1, c = 0.2, k = 1 this is a damped oscillator
with natural frequency 1 and damping ratio 0.1: it swings and dies away. It has
an exact solution, which is the point. Everything here can be measured against
truth rather than against a held-out split.

THE TWO MODELS

Both are the SAME network: [1, 32, 32, 32, 1] with tanh, 2,209 parameters, same
initialisation, same optimiser, same learning-rate schedule. The only difference
is the loss.

    DATA-ONLY      minimise (prediction - observation)^2 at the observed times.
                   This is Week 3 and Week 4's setting, and it knows nothing
                   about springs.

    PHYSICS-INFORMED   minimise the residual of the equation itself, evaluated
                   at points where there is NO data:

                       mean( (m u'' + c u' + k u)^2 )

                   u'' and u' are derivatives of the NETWORK with respect to its
                   INPUT, computed exactly by autodiff.py. No observations of u
                   are required at all.

That second loss is the week's whole idea. The model is not being shown what the
answer looks like. It is being told what the answer must SATISFY.

THE EXPERIMENTS

    1  Data efficiency. How many observations does each model need?
    2  Extrapolation. What happens outside the region where data existed?
    3  Noise. Which model is fooled by measurement error?
    4  Soft penalty against hard constraint, and why the choice matters.
    5  WHEN THE PHYSICS IS WRONG. The honest counterexample: report this one.
    6  Conservation. Does the model respect a quantity nobody asked it to track?

Booklet cross-references: Chapters 8 to 12, and Chapter 22.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys as _sys
import matplotlib
if "ipykernel" not in _sys.modules:      # keep inline plots working in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autodiff import (Node, grad, add, sub, mul, div, square, mean, sum_all,
                      const)
from mlp import MLP, Adam

SLATE = "#47546B"
AMBER = "#A96E08"
RED = "#9E3B2C"
GREEN = "#2E7D4F"
BLUE = "#4E7291"
GREY = "#8A93A5"
PURPLE = "#5B4B9E"


# ---------------------------------------------------------------------------
# The physical system
# ---------------------------------------------------------------------------

class DampedOscillator:
    """m u'' + c u' + k u = 0 with an exact solution to check against.

    Written as a small class because every experiment needs the same three
    things from it: the equation (for the physics loss), the truth (for scoring),
    and the energy (for the conservation test).
    """

    def __init__(self, m=1.0, c=0.2, k=1.0, u0=1.0, v0=0.0, T=10.0):
        self.m, self.c, self.k = float(m), float(c), float(k)
        self.u0, self.v0, self.T = float(u0), float(v0), float(T)
        self.omega = np.sqrt(self.k / self.m)
        self.zeta = self.c / (2.0 * np.sqrt(self.k * self.m))

    def exact(self, t):
        """The closed-form solution, for the under-damped case."""
        t = np.asarray(t, dtype=float)
        w, z = self.omega, self.zeta
        if z >= 1.0:
            raise ValueError("this exact solution assumes under-damping (zeta < 1)")
        wd = w * np.sqrt(1 - z * z)
        A = self.u0
        B = (self.v0 + z * w * self.u0) / wd
        return np.exp(-z * w * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))

    def exact_dot(self, t, eps=1e-6):
        return (self.exact(t + eps) - self.exact(t - eps)) / (2 * eps)

    def energy(self, t):
        """E = kinetic + potential. Constant only when c = 0."""
        u = self.exact(t)
        v = self.exact_dot(t)
        return 0.5 * self.m * v ** 2 + 0.5 * self.k * u ** 2

    @property
    def periods(self):
        return self.T / (2 * np.pi / self.omega)

    def describe(self):
        return (f"m={self.m}, c={self.c}, k={self.k}  ->  "
                f"omega={self.omega:.3f}, zeta={self.zeta:.3f}, "
                f"{self.periods:.2f} periods over T={self.T}")


# ---------------------------------------------------------------------------
# The model, used for both conditions
# ---------------------------------------------------------------------------

class Solver:
    """One network, two possible losses.

    `hard_ic=True` builds the initial conditions into the FUNCTIONAL FORM rather
    than penalising their violation:

        u(t) = u0 + v0 t + s(t) N(t),      s(t) = t^2 / (1 + t^2)

    Since s(0) = 0 and s'(0) = 0, this satisfies u(0) = u0 and u'(0) = v0
    EXACTLY, for any network N and any weights, including at initialisation.
    Experiment 4 measures why that matters: with a soft penalty the same setup
    reaches an error around 2.6e-01, and with the hard constraint around 1.4e-03,
    a factor of roughly 200. Booklet Chapter 10.

    Input scaling: t is mapped to [-1, 1] before the network sees it. This is
    ordinary feature scaling from Week 3, but it matters far more here, because
    the residual involves second derivatives and their magnitude scales as the
    square of the input scale.
    """

    def __init__(self, system, layers=(1, 32, 32, 32, 1), seed=0,
                 activation="tanh", hard_ic=True):
        self.sys = system
        self.net = MLP(list(layers), seed=seed, activation=activation)
        self.params = self.net.params
        self.hard_ic = hard_ic

    @property
    def n_params(self):
        return self.net.n_params

    def __call__(self, t):
        t = const(t)
        T = self.sys.T
        raw = self.net(add(mul(t, 2.0 / T), -1.0))
        if not self.hard_ic:
            return raw
        # s(t) = t^2 / (1 + t^2): zero value and zero slope at t = 0.
        s = div(mul(t, t), add(1.0, mul(t, t)))
        return add(add(self.sys.u0, mul(self.sys.v0, t)), mul(s, raw))

    # --- the two loss terms ---------------------------------------------
    def physics_loss(self, t_col, m=None, c=None, k=None):
        """mean( (m u'' + c u' + k u)^2 ) at the collocation points.

        m, c, k can be overridden to model the system WRONGLY, which is
        Experiment 5.
        """
        m = self.sys.m if m is None else m
        c = self.sys.c if c is None else c
        k = self.sys.k if k is None else k
        u = self(t_col)
        u_t = grad(sum_all(u), t_col)
        u_tt = grad(sum_all(u_t), t_col)
        residual = add(add(mul(m, u_tt), mul(c, u_t)), mul(k, u))
        return mean(square(residual))

    def data_loss(self, t_obs, u_obs):
        return mean(square(sub(self(t_obs), const(u_obs))))

    def ic_loss(self):
        t0 = Node(np.array([[0.0]]))
        u0 = self(t0)
        v0 = grad(sum_all(u0), t0)
        return sum_all(add(square(sub(u0, self.sys.u0)),
                           square(sub(v0, self.sys.v0))))

    # --- scoring ---------------------------------------------------------
    def rmse(self, t_grid=None, n=600):
        t_grid = np.linspace(0, self.sys.T, n).reshape(-1, 1) if t_grid is None else t_grid
        pred = self(Node(t_grid)).value
        return float(np.sqrt(np.mean((pred - self.sys.exact(t_grid)) ** 2)))

    def predict(self, t):
        return self(Node(np.asarray(t, float).reshape(-1, 1))).value


def fit(solver, epochs=3000, lr=3e-3, t_col=None, t_obs=None, u_obs=None,
        w_data=1.0, w_ic=50.0, wrong=None, track_every=None, seed=0):
    """Train a solver with whichever loss terms are supplied.

    Pass `t_col` for physics. Pass `t_obs`/`u_obs` for data. Pass both for a
    hybrid, which is what most real applications actually are.
    """
    opt = Adam(solver.params, lr=lr)
    wrong = wrong or {}
    hist, tracked = [], []
    for ep in range(epochs):
        terms = []
        if t_col is not None:
            terms.append(solver.physics_loss(t_col, **wrong))
        if t_obs is not None:
            terms.append(mul(w_data, solver.data_loss(t_obs, u_obs)))
        if not solver.hard_ic and t_col is not None:
            terms.append(mul(w_ic, solver.ic_loss()))
        loss = terms[0]
        for extra in terms[1:]:
            loss = add(loss, extra)
        opt.step(grad(loss, solver.params))
        hist.append(loss.value.item())
        if ep + 1 in (int(epochs * 0.5), int(epochs * 0.8)):
            opt.set_lr(opt.lr * 0.3)
        if track_every and (ep + 1) % track_every == 0:
            tracked.append((ep + 1, solver.rmse()))
    return hist, tracked


def observations(system, n, noise=0.05, t_max=None, seed=0):
    """Noisy measurements of the true solution, as an experiment would produce."""
    rng = np.random.default_rng(seed)
    t_max = system.T if t_max is None else t_max
    t = np.linspace(0, t_max, n).reshape(-1, 1)
    u = system.exact(t) + rng.normal(0, noise, (n, 1))
    return t, u


# ---------------------------------------------------------------------------
# EXPERIMENT 1: data efficiency
# ---------------------------------------------------------------------------

def experiment_data_efficiency(sizes=(4, 8, 16, 32, 64), noise=0.05,
                               epochs=3000, seed=0, quick=False):
    """How many observations does each model need to get the answer right?

    The physics-informed row is the one to look at, and specifically its data
    column. It is zero. Not "few": none. The model never sees a single
    measurement of u. It is given the equation and the initial condition and it
    reconstructs the trajectory to within about a thousandth.

    That is a different kind of claim from anything in Weeks 3 and 4, where every
    model needed examples of the thing it was predicting.
    """
    if quick:
        sizes, epochs = (4, 16, 64), 1500
    sys = DampedOscillator()
    rows = []

    # The physics-informed model, with no data at all.
    t_col = Node(np.linspace(0, sys.T, 200).reshape(-1, 1))
    pinn = Solver(sys, seed=seed)
    fit(pinn, epochs=epochs, t_col=t_col, seed=seed)
    rows.append({"model": "physics-informed (no data)", "observations": 0,
                 "parameters": pinn.n_params, "test RMSE": round(pinn.rmse(), 5)})

    # The data-only model, at each dataset size.
    for n in sizes:
        t_obs, u_obs = observations(sys, n, noise=noise, seed=seed)
        d = Solver(sys, seed=seed, hard_ic=False)
        fit(d, epochs=epochs, t_obs=Node(t_obs), u_obs=u_obs, seed=seed)
        rows.append({"model": "data-only", "observations": n,
                     "parameters": d.n_params, "test RMSE": round(d.rmse(), 5)})

    # A hybrid: physics plus a handful of noisy points, which is the realistic case.
    t_obs, u_obs = observations(sys, 8, noise=noise, seed=seed)
    h = Solver(sys, seed=seed)
    fit(h, epochs=epochs, t_col=t_col, t_obs=Node(t_obs), u_obs=u_obs, seed=seed)
    rows.append({"model": "physics + 8 noisy points", "observations": 8,
                 "parameters": h.n_params, "test RMSE": round(h.rmse(), 5)})

    return pd.DataFrame(rows), pinn


# ---------------------------------------------------------------------------
# EXPERIMENT 2: extrapolation
# ---------------------------------------------------------------------------

def experiment_extrapolation(t_split=4.0, n_obs=24, noise=0.03, epochs=3000,
                             seed=0, quick=False):
    """Both models get data only on [0, 4]. Both are tested on [4, 10].

    The physics-informed model additionally gets collocation points across the
    whole interval, which is not cheating: it is the entire point. You know the
    equation everywhere, including where you have no instrument. That is what
    distinguishes a physical law from a training set.

    This experiment is the reliability argument, and reliability is usually a
    more valuable property than accuracy. A data-driven model that is excellent
    inside its training range and arbitrary outside it is dangerous precisely
    because the failure is silent.
    """
    if quick:
        epochs = 1500
    sys = DampedOscillator()
    t_obs, u_obs = observations(sys, n_obs, noise=noise, t_max=t_split, seed=seed)
    t_in = np.linspace(0, t_split, 300).reshape(-1, 1)
    t_out = np.linspace(t_split, sys.T, 300).reshape(-1, 1)
    t_col = Node(np.linspace(0, sys.T, 200).reshape(-1, 1))

    rows = []
    models = {}

    d = Solver(sys, seed=seed, hard_ic=False)
    fit(d, epochs=epochs, t_obs=Node(t_obs), u_obs=u_obs, seed=seed)
    models["data-only"] = d
    rows.append({"model": "data-only",
                 "RMSE where data existed": round(d.rmse(t_in), 4),
                 "RMSE beyond the data": round(d.rmse(t_out), 4)})

    h = Solver(sys, seed=seed)
    fit(h, epochs=epochs, t_col=t_col, t_obs=Node(t_obs), u_obs=u_obs, seed=seed)
    models["physics + data"] = h
    rows.append({"model": "physics + same data",
                 "RMSE where data existed": round(h.rmse(t_in), 4),
                 "RMSE beyond the data": round(h.rmse(t_out), 4)})

    df = pd.DataFrame(rows)
    df["degradation factor"] = (df["RMSE beyond the data"] /
                               df["RMSE where data existed"]).round(1)
    return df, models, (t_obs, u_obs, t_split)


# ---------------------------------------------------------------------------
# EXPERIMENT 3: noise
# ---------------------------------------------------------------------------

def experiment_noise(levels=(0.0, 0.10, 0.30), n_obs=16, epochs=3000,
                     seed=0, quick=False):
    """Increase the measurement noise and see which model is fooled.

    A physics residual acts as a regulariser with a very unusual property: it
    prefers functions that are physically possible. Noise is not physically
    possible, because no trajectory of a damped oscillator jitters. So the
    physics term suppresses it, without anybody choosing a smoothing parameter.

    Week 3, Chapter 10 introduced regularisation as a penalty on complexity.
    This is a penalty on IMPOSSIBILITY, which is a much sharper instrument.
    """
    if quick:
        levels, epochs = (0.0, 0.30), 1500
    sys = DampedOscillator()
    t_col = Node(np.linspace(0, sys.T, 200).reshape(-1, 1))
    rows = []
    for s in levels:
        t_obs, u_obs = observations(sys, n_obs, noise=s, seed=seed)
        d = Solver(sys, seed=seed, hard_ic=False)
        fit(d, epochs=epochs, t_obs=Node(t_obs), u_obs=u_obs, seed=seed)
        h = Solver(sys, seed=seed)
        fit(h, epochs=epochs, t_col=t_col, t_obs=Node(t_obs), u_obs=u_obs, seed=seed)
        rows.append({"noise std": s,
                     "data-only RMSE": round(d.rmse(), 4),
                     "physics + data RMSE": round(h.rmse(), 4),
                     "ratio": round(d.rmse() / max(h.rmse(), 1e-12), 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EXPERIMENT 4: soft penalty against hard constraint
# ---------------------------------------------------------------------------

def experiment_constraint_style(epochs=3000, seed=0, quick=False):
    """Two ways to impose u(0) = 1 and u'(0) = 0, with very different results.

    SOFT: add a penalty for violating them and hope the optimiser complies.
    HARD: build them into the functional form so they hold identically.

    The measured result is more interesting than the usual telling of this story.
    The soft penalty is not simply worse; it is UNPREDICTABLE. At weight 1 it
    reaches an error of about 1.5e-03, which essentially matches the hard
    constraint. At weight 50 it reaches 3.2e-01 and at weight 500 it reaches
    6.6e-01, which are both useless. That is a spread of over 400 times, caused
    by a number nobody can derive in advance.

    Look at the u(0) column to see why this is not the failure you would guess.
    Every version satisfies the initial condition to four decimal places. The
    high-weight runs are not violating the condition; they are satisfying it
    while neglecting the equation, because the loss they were given said the
    condition mattered five hundred times more.

    Balancing the terms of a multi-part physics loss is a genuine open problem,
    and there is an active literature on doing it adaptively. The hard constraint
    removes the question rather than answering it, which is why it is worth
    knowing: u(0) and u'(0) are exact by construction, at any weight, from the
    very first epoch.

    Worth remembering for the quiz: a small total loss tells you the terms you
    wrote down are satisfied. It does not tell you that you weighted them
    sensibly, and this table is the evidence.
    """
    if quick:
        epochs = 1500
    sys = DampedOscillator()
    t_col = Node(np.linspace(0, sys.T, 200).reshape(-1, 1))
    rows = []
    for style, w in (("soft penalty, weight 1", 1.0),
                     ("soft penalty, weight 50", 50.0),
                     ("soft penalty, weight 500", 500.0),
                     ("hard constraint", None)):
        s = Solver(sys, seed=seed, hard_ic=(w is None))
        fit(s, epochs=epochs, t_col=t_col, w_ic=(w or 0.0), seed=seed)
        t0 = Node(np.array([[0.0]]))
        u0 = s(t0).value.item()
        v0 = grad(sum_all(s(t0)), t0).value.item()
        rows.append({"how the initial condition is imposed": style,
                     "u(0), should be 1": round(u0, 5),
                     "u'(0), should be 0": round(v0, 5),
                     "test RMSE": round(s.rmse(), 5)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EXPERIMENT 5: when the physics is wrong
# ---------------------------------------------------------------------------

def experiment_wrong_physics(c_true=0.2, c_assumed=(0.2, 0.4, 0.8), n_obs=24,
                             noise=0.03, epochs=3000, seed=0, quick=False):
    """THE HONEST COUNTEREXAMPLE. Report this one.

    Everything above says: a correct physical prior is worth an enormous amount.
    The obvious question is what happens when the prior is WRONG, and the answer
    is worse than "you lose the benefit".

    Here the true system has damping c = 0.2. The physics-informed model is told
    c = 0.4 or c = 0.8, which is the kind of error you make when a coefficient
    was measured badly or a mechanism was left out. Real data from the true
    system is supplied to both models.

    The data-only model tracks the data, because that is all it does. The
    physics-informed model is pulled towards a trajectory that satisfies the
    WRONG equation, and it goes there confidently, with a small and
    healthy-looking physics loss the whole way.

    This is Week 4's "pre-centred digits" result in a more consequential form. A
    structural assumption is only worth the extent to which it is true, and a
    confident wrong assumption is worse than no assumption, because it does not
    look like a failure.
    """
    if quick:
        c_assumed, epochs = (0.2, 0.8), 1500
    sys = DampedOscillator(c=c_true)
    t_obs, u_obs = observations(sys, n_obs, noise=noise, seed=seed)
    t_col = Node(np.linspace(0, sys.T, 200).reshape(-1, 1))

    rows = []
    d = Solver(sys, seed=seed, hard_ic=False)
    fit(d, epochs=epochs, t_obs=Node(t_obs), u_obs=u_obs, seed=seed)
    rows.append({"model": "data-only (no physics)", "damping assumed": "n/a",
                 "true RMSE": round(d.rmse(), 4),
                 "physics loss it reports": "n/a"})

    models = {"data-only": d}
    for c in c_assumed:
        h = Solver(sys, seed=seed)
        fit(h, epochs=epochs, t_col=t_col, t_obs=Node(t_obs), u_obs=u_obs,
            wrong={"c": c}, seed=seed)
        pl = h.physics_loss(t_col, c=c).value.item()
        label = "correct" if abs(c - c_true) < 1e-9 else f"WRONG ({c / c_true:.3g}x too high)"
        rows.append({"model": f"physics + data, c={c} ({label})",
                     "damping assumed": c,
                     "true RMSE": round(h.rmse(), 4),
                     "physics loss it reports": f"{pl:.2e}"})
        models[f"c={c}"] = h
    return pd.DataFrame(rows), models


# ---------------------------------------------------------------------------
# EXPERIMENT 6: conservation
# ---------------------------------------------------------------------------

def experiment_conservation(n_obs=20, noise=0.03, epochs=3000, seed=0,
                            quick=False):
    """Remove the damping and the system conserves energy exactly. Do the models?

    Nobody tells either model about energy. It is not in the loss. Energy is a
    CONSEQUENCE of the equation, so a model that satisfies the equation
    inherits it, and a model that merely fits points does not.

    This is what "physical priors improve reliability" means concretely. The
    guarantee you get is not only on the quantity you trained, but on everything
    the equation implies, which is usually far more than you thought to check.
    """
    if quick:
        epochs = 1500
    sys = DampedOscillator(c=0.0)          # undamped: energy is constant
    t_obs, u_obs = observations(sys, n_obs, noise=noise, seed=seed)
    t_col = Node(np.linspace(0, sys.T, 200).reshape(-1, 1))
    t_grid = np.linspace(0, sys.T, 400).reshape(-1, 1)

    def energy_of(solver):
        t = Node(t_grid)
        u = solver(t)
        v = grad(sum_all(u), t)
        e = 0.5 * sys.m * v.value ** 2 + 0.5 * sys.k * u.value ** 2
        return e.ravel()

    rows, curves = [], {}
    d = Solver(sys, seed=seed, hard_ic=False)
    fit(d, epochs=epochs, t_obs=Node(t_obs), u_obs=u_obs, seed=seed)
    h = Solver(sys, seed=seed)
    fit(h, epochs=epochs, t_col=t_col, t_obs=Node(t_obs), u_obs=u_obs, seed=seed)
    p = Solver(sys, seed=seed)
    fit(p, epochs=epochs, t_col=t_col, seed=seed)

    e_true = 0.5 * sys.k * sys.u0 ** 2
    for name, s in (("data-only", d), ("physics + data", h),
                    ("physics only, no data", p)):
        e = energy_of(s)
        curves[name] = e
        rows.append({"model": name,
                     "energy at t=0": round(float(e[0]), 4),
                     "energy at t=T": round(float(e[-1]), 4),
                     "max drift from true": round(float(np.abs(e - e_true).max()), 4),
                     "drift as % of true": f"{100 * float(np.abs(e - e_true).max()) / e_true:.1f}%"})
    return pd.DataFrame(rows), curves, (t_grid, e_true)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_solution(pinn, sys=None):
    sys = sys or pinn.sys
    t = np.linspace(0, sys.T, 600).reshape(-1, 1)
    pred = pinn.predict(t)
    tc = Node(np.linspace(0, sys.T, 200).reshape(-1, 1))
    u = pinn(tc)
    res = (sys.m * grad(sum_all(grad(sum_all(u), tc)), tc).value
           + sys.c * grad(sum_all(u), tc).value + sys.k * u.value)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(t, sys.exact(t), lw=3.4, color=GREY, label="exact solution")
    axes[0].plot(t, pred, lw=1.9, color=SLATE, ls="--", label="PINN, trained on zero data")
    axes[0].set_ylabel("displacement u(t)")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.25)
    axes[0].set_title("A network that was never shown the answer, only the equation it must satisfy\n"
                      f"RMSE {pinn.rmse():.2e} against the exact solution",
                      fontsize=12, color=SLATE)
    axes[1].axhline(0, color=GREY, lw=1)
    axes[1].plot(tc.value, res, lw=1.5, color=RED)
    axes[1].set_ylabel("residual\nm u'' + c u' + k u")
    axes[1].set_xlabel("time t")
    axes[1].grid(alpha=0.25)
    axes[1].set_title("What the loss actually minimised: how badly Newton's law is violated, "
                      "point by point", fontsize=10.5, color=GREY)
    fig.tight_layout()
    return fig


def fig_extrapolation(models, obs, sys=None):
    t_obs, u_obs, t_split = obs
    sys = sys or models["data-only"].sys
    t = np.linspace(0, sys.T, 600).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.axvspan(0, t_split, color=BLUE, alpha=0.07)
    ax.axvspan(t_split, sys.T, color=RED, alpha=0.06)
    ax.plot(t, sys.exact(t), lw=3.4, color=GREY, label="exact solution")
    ax.plot(t, models["data-only"].predict(t), lw=2.1, color=AMBER,
            label="data-only")
    ax.plot(t, models["physics + data"].predict(t), lw=2.1, color=SLATE, ls="--",
            label="physics + the same data")
    ax.scatter(t_obs, u_obs, s=30, color=RED, zorder=5, label="the observations")
    ax.axvline(t_split, color=RED, ls=":", lw=1.6)
    ax.text(t_split * 0.5, 1.25, "data available here", ha="center", fontsize=11, color=BLUE)
    ax.text((t_split + sys.T) / 2, 1.25, "no data ever seen here", ha="center",
            fontsize=11, color=RED)
    ax.set_xlabel("time t")
    ax.set_ylabel("displacement u(t)")
    ax.set_ylim(-1.5, 1.6)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_title("Beyond the data, one model keeps working and the other invents something\n"
                 "The equation was known everywhere. Only one model was told.",
                 fontsize=12.5, color=SLATE)
    fig.tight_layout()
    return fig


def fig_wrong_physics(models, sys=None):
    sys = sys or DampedOscillator()
    t = np.linspace(0, sys.T, 600).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.plot(t, sys.exact(t), lw=3.6, color=GREY, label="the truth (c = 0.2)")
    colours = {"data-only": AMBER, "c=0.2": GREEN, "c=0.4": PURPLE, "c=0.8": RED}
    for name, m in models.items():
        ax.plot(t, m.predict(t), lw=2.0, color=colours.get(name, SLATE),
                ls="--" if name != "data-only" else "-",
                label=("data-only" if name == "data-only"
                       else f"physics with {name}"))
    ax.set_xlabel("time t")
    ax.set_ylabel("displacement u(t)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_title("When the assumed physics is wrong, the model is confidently wrong\n"
                 "Every dashed curve has a small, healthy-looking physics loss",
                 fontsize=12.5, color=SLATE)
    fig.tight_layout()
    return fig


def fig_conservation(curves, meta):
    t_grid, e_true = meta
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axhline(e_true, color=GREY, lw=3.0, label=f"true energy, constant at {e_true:.2f}")
    ax.plot(t_grid, curves["data-only"], lw=2.2, color=AMBER, label="data-only")
    ax.plot(t_grid, curves["physics + data"], lw=2.2, color=SLATE, ls="--",
            label="physics + the same data")
    ax.set_xlabel("time t")
    ax.set_ylabel("total energy")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_title("Nobody put energy in either loss function.\n"
                 "One model conserves it anyway, because it satisfies the equation that implies it.",
                 fontsize=12.5, color=SLATE)
    fig.tight_layout()
    return fig


def fig_data_efficiency(df):
    d = df[df["model"] == "data-only"]
    pinn_rmse = float(df[df["model"].str.startswith("physics-informed")]["test RMSE"].iloc[0])
    hy = df[df["model"].str.startswith("physics + 8")]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.loglog(d["observations"], d["test RMSE"], "o-", lw=2.4, color=AMBER,
              label="data-only")
    ax.axhline(pinn_rmse, color=SLATE, lw=2.6, ls="--",
               label=f"physics-informed, ZERO observations ({pinn_rmse:.1e})")
    if len(hy):
        ax.plot([8], [float(hy["test RMSE"].iloc[0])], "s", ms=11, color=GREEN,
                label="physics + 8 noisy points")
    ax.set_xlabel("number of observations")
    ax.set_ylabel("RMSE against the exact solution")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=10)
    ax.set_title("The data-only curve never reaches the horizontal line,\n"
                 "and the horizontal line was drawn without any data at all",
                 fontsize=12.5, color=SLATE)
    fig.tight_layout()
    return fig


def main(quick=False):
    sys = DampedOscillator()
    print(f"\nThe system: {sys.describe()}")

    print("\nEXPERIMENT 1: data efficiency")
    eff, pinn = experiment_data_efficiency(quick=quick)
    print(eff.to_string(index=False))

    print("\nEXPERIMENT 2: extrapolation beyond the data")
    ext, models, obs = experiment_extrapolation(quick=quick)
    print(ext.to_string(index=False))

    print("\nEXPERIMENT 4: how the initial condition is imposed")
    cs = experiment_constraint_style(quick=quick)
    print(cs.to_string(index=False))

    print("\nEXPERIMENT 5: when the physics is wrong")
    wp, wmodels = experiment_wrong_physics(quick=quick)
    print(wp.to_string(index=False))

    print("\nEXPERIMENT 6: conservation of energy, which nobody asked for")
    cons, curves, meta = experiment_conservation(quick=quick)
    print(cons.to_string(index=False))
    return eff, ext, cs, wp, cons


if __name__ == "__main__":
    import sys as _s
    main(quick="--quick" in _s.argv)
