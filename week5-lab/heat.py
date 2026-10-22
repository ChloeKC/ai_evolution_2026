"""
heat.py, the same idea in two dimensions: a partial differential equation.

The Evolution of Machine Learning and AI, Week 5 Lab
South East Technological University

THE PROBLEM

Heat flowing along a thin insulated rod of length 1. The temperature u(x, t)
obeys the heat equation, which is the canonical parabolic PDE:

    du/dt = alpha d^2u/dx^2

with the rod held at zero at both ends and starting from a single smooth bump:

    u(0, t) = u(1, t) = 0          the ends are clamped to zero
    u(x, 0) = sin(pi x)            the initial temperature profile

This has an exact solution, which again is the point:

    u(x, t) = exp(-alpha pi^2 t) sin(pi x)

The shape stays a sine and the amplitude decays exponentially. With alpha = 0.2
the amplitude falls to about 0.14 of its starting value by t = 1.

WHY THIS FILE EXISTS ALONGSIDE pinn.py

Three things change when you move from an ODE to a PDE, and all three are worth
meeting once.

    1  The network now takes TWO inputs, so you need derivatives with respect to
       each separately: du/dt and d^2u/dx^2. The engine handles it by keeping x
       and t as separate nodes and joining them with `concat`, so `grad` can
       target one without the other.

    2  The conditions are no longer just at a point. There is an initial
       condition along the whole bottom edge and boundary conditions along both
       sides. That is three constraint families rather than two numbers, which is
       why hard constraints become more attractive rather than less.

    3  The collocation points now fill an AREA. A 40 by 40 grid is 1,600 points,
       and in three spatial dimensions plus time the same grid spacing would be
       2.56 million. This is where the honest limitation of PINNs starts to show,
       and Chapter 13 discusses it.

Booklet cross-reference: Chapter 13.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys as _sys
import matplotlib
if "ipykernel" not in _sys.modules:      # keep inline plots working in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autodiff import (Node, grad, add, sub, mul, square, mean, sum_all, sin,
                      concat, const)
from mlp import MLP, Adam

SLATE = "#47546B"
AMBER = "#A96E08"
RED = "#9E3B2C"
GREY = "#8A93A5"


class HeatEquation:
    def __init__(self, alpha=0.2, T=1.0, L=1.0):
        self.alpha, self.T, self.L = float(alpha), float(T), float(L)

    def exact(self, x, t):
        return np.exp(-self.alpha * np.pi ** 2 * t) * np.sin(np.pi * x)

    def describe(self):
        decay = np.exp(-self.alpha * np.pi ** 2 * self.T)
        return (f"alpha={self.alpha}, rod length {self.L}, t up to {self.T}. "
                f"Amplitude falls to {decay:.3f} of its start.")


class HeatSolver:
    """u(x, t) = sin(pi x) [1 + t N(x, t)], which satisfies everything by construction.

    Check the three conditions by hand, because this is the clearest example in
    the lab of a hard constraint doing real work:

        at t = 0:  the bracket is 1, so u = sin(pi x). The initial condition
                   holds exactly, for any weights.
        at x = 0:  sin(0) = 0, so u = 0. The left boundary holds exactly.
        at x = 1:  sin(pi) = 0, so u = 0. The right boundary holds exactly.

    So the loss has exactly ONE term, the PDE residual. No weighting to balance,
    no penalties to tune, and no possibility of the optimiser trading away a
    boundary condition to buy a slightly smaller residual. Compare
    `experiment_constraint_style` in pinn.py, where getting that trade wrong cost
    a factor of 400 in accuracy.
    """

    def __init__(self, system, layers=(2, 32, 32, 32, 1), seed=0, hard=True):
        self.sys = system
        self.net = MLP(list(layers), seed=seed, activation="tanh")
        self.params = self.net.params
        self.hard = hard

    @property
    def n_params(self):
        return self.net.n_params

    def __call__(self, x, t):
        x, t = const(x), const(t)
        # scale both inputs to [-1, 1] before the network sees them
        xn = add(mul(x, 2.0 / self.sys.L), -1.0)
        tn = add(mul(t, 2.0 / self.sys.T), -1.0)
        raw = self.net(concat([xn, tn], axis=1))
        if not self.hard:
            return raw
        envelope = sin(mul(np.pi / self.sys.L, x))
        return mul(envelope, add(1.0, mul(t, raw)))

    def residual(self, x, t):
        """du/dt - alpha d^2u/dx^2, evaluated pointwise.

        Note that x and t are separate nodes. That is what allows `grad` to ask
        for the derivative in one direction only, which is the whole reason the
        inputs are not simply one array of shape (N, 2).
        """
        u = self(x, t)
        u_t = grad(sum_all(u), t)
        u_x = grad(sum_all(u), x)
        u_xx = grad(sum_all(u_x), x)
        return sub(u_t, mul(self.sys.alpha, u_xx))

    def physics_loss(self, x, t, alpha=None):
        a = self.sys.alpha if alpha is None else alpha
        u = self(x, t)
        u_t = grad(sum_all(u), t)
        u_x = grad(sum_all(u), x)
        u_xx = grad(sum_all(u_x), x)
        return mean(square(sub(u_t, mul(a, u_xx))))

    def data_loss(self, x, t, u_obs):
        return mean(square(sub(self(x, t), const(u_obs))))

    def boundary_loss(self, n=40, seed=0):
        """Only needed when hard=False. Three families, all as penalties."""
        rng = np.random.default_rng(seed)
        xi = Node(rng.uniform(0, self.sys.L, (n, 1)))
        t0 = Node(np.zeros((n, 1)))
        ic = mean(square(sub(self(xi, t0),
                             np.sin(np.pi * xi.value / self.sys.L))))
        tb = Node(rng.uniform(0, self.sys.T, (n, 1)))
        left = mean(square(self(Node(np.zeros((n, 1))), tb)))
        right = mean(square(self(Node(np.full((n, 1), self.sys.L)), tb)))
        return add(ic, add(left, right))

    def rmse(self, nx=60, nt=60):
        x = np.linspace(0, self.sys.L, nx)
        t = np.linspace(0, self.sys.T, nt)
        X, T = np.meshgrid(x, t, indexing="ij")
        xf = X.reshape(-1, 1)
        tf = T.reshape(-1, 1)
        pred = self(Node(xf), Node(tf)).value
        return float(np.sqrt(np.mean((pred - self.sys.exact(xf, tf)) ** 2)))

    def field(self, nx=60, nt=60):
        x = np.linspace(0, self.sys.L, nx)
        t = np.linspace(0, self.sys.T, nt)
        X, T = np.meshgrid(x, t, indexing="ij")
        pred = self(Node(X.reshape(-1, 1)), Node(T.reshape(-1, 1))).value
        return x, t, pred.reshape(nx, nt)


def collocation(system, n_side=32, seed=0):
    """A grid of interior points. n_side^2 of them, which is the cost worth noticing."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, system.L, n_side)
    t = np.linspace(0, system.T, n_side)
    X, T = np.meshgrid(x, t, indexing="ij")
    return Node(X.reshape(-1, 1)), Node(T.reshape(-1, 1))


def observations(system, n, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, system.L, (n, 1))
    t = rng.uniform(0, system.T, (n, 1))
    u = system.exact(x, t) + rng.normal(0, noise, (n, 1))
    return Node(x), Node(t), u


def fit_heat(solver, epochs=2000, lr=3e-3, col=None, obs=None, w_bc=10.0,
             alpha=None):
    opt = Adam(solver.params, lr=lr)
    hist = []
    for ep in range(epochs):
        terms = []
        if col is not None:
            terms.append(solver.physics_loss(col[0], col[1], alpha=alpha))
            if not solver.hard:
                terms.append(mul(w_bc, solver.boundary_loss()))
        if obs is not None:
            terms.append(solver.data_loss(obs[0], obs[1], obs[2]))
        loss = terms[0]
        for extra in terms[1:]:
            loss = add(loss, extra)
        opt.step(grad(loss, solver.params))
        hist.append(loss.value.item())
        if ep + 1 in (int(epochs * 0.5), int(epochs * 0.8)):
            opt.set_lr(opt.lr * 0.3)
    return hist


def experiment_heat(n_obs=(8, 32, 128), noise=0.03, epochs=2000, seed=0,
                    quick=False):
    """The same comparison as pinn.py, on a PDE.

    The physics-informed row again uses no observations. It is given the equation,
    and the conditions are built into its functional form, so the only thing it
    minimises is the extent to which it violates the heat equation.
    """
    if quick:
        n_obs, epochs = (8, 128), 1200
    sysm = HeatEquation()
    col = collocation(sysm, n_side=32, seed=seed)
    rows = []

    p = HeatSolver(sysm, seed=seed)
    fit_heat(p, epochs=epochs, col=col)
    rows.append({"model": "physics-informed (no data)", "observations": 0,
                 "parameters": p.n_params, "RMSE": round(p.rmse(), 5)})

    for n in n_obs:
        obs = observations(sysm, n, noise=noise, seed=seed)
        d = HeatSolver(sysm, seed=seed, hard=False)
        fit_heat(d, epochs=epochs, obs=obs)
        rows.append({"model": "data-only", "observations": n,
                     "parameters": d.n_params, "RMSE": round(d.rmse(), 5)})
    return pd.DataFrame(rows), p


def fig_heat(solver):
    sysm = solver.sys
    x, t, pred = solver.field()
    X, T = np.meshgrid(x, t, indexing="ij")
    truth = sysm.exact(X, T)
    err = np.abs(pred - truth)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    for ax, field, title, cm in (
        (axes[0], truth, "exact solution", "inferno"),
        (axes[1], pred, "the PINN, trained on zero data", "inferno"),
        (axes[2], err, f"absolute error (max {err.max():.1e})", "viridis"),
    ):
        im = ax.imshow(field.T, origin="lower", aspect="auto", cmap=cm,
                       extent=[0, sysm.L, 0, sysm.T])
        ax.set_xlabel("position x")
        ax.set_title(title, fontsize=11, color=SLATE)
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].set_ylabel("time t")
    fig.suptitle("Heat flowing along a rod, solved by minimising violation of the heat equation",
                 fontsize=12.5, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


def fig_heat_slices(solver):
    sysm = solver.sys
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.linspace(0, sysm.L, 200).reshape(-1, 1)
    for i, tv in enumerate((0.0, 0.15, 0.4, 1.0)):
        tt = np.full_like(xs, tv)
        col = plt.cm.inferno(0.15 + 0.7 * i / 3)
        ax.plot(xs, sysm.exact(xs, tt), lw=3.2, color=GREY,
                label="exact" if i == 0 else None)
        ax.plot(xs, solver(Node(xs), Node(tt)).value, lw=2.0, ls="--", color=col,
                label=f"PINN at t = {tv}")
    ax.set_xlabel("position along the rod, x")
    ax.set_ylabel("temperature u")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_title("The bump flattens and never changes shape, which is what the equation says it must do",
                 fontsize=12, color=SLATE)
    fig.tight_layout()
    return fig


def main(quick=False):
    sysm = HeatEquation()
    print(f"\nThe system: {sysm.describe()}")
    print("\nThe hard constraint u(x,t) = sin(pi x)[1 + t N(x,t)] satisfies")
    print("the initial condition and both boundaries EXACTLY, for any weights,")
    print("so the loss has exactly one term: the PDE residual.")
    df, p = experiment_heat(quick=quick)
    print()
    print(df.to_string(index=False))
    return df, p


if __name__ == "__main__":
    import sys as _s
    main(quick="--quick" in _s.argv)
