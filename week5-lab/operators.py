"""
operators.py, learning the solution MAP rather than one solution.

The Evolution of Machine Learning and AI, Week 5 Lab
South East Technological University

THE LIMITATION THIS FIXES

A PINN is trained for one problem. One equation, one set of coefficients, one
initial condition. Change the damping from 0.2 to 0.25 and the trained network is
worthless: you start again, and on this laptop that is another twenty seconds.
On a real three-dimensional PDE it is another several hours.

That is often unacceptable. Design optimisation, uncertainty quantification,
control and inverse problems all involve solving the SAME equation thousands of
times with different coefficients. Paying full price each time is the bottleneck.

A neural operator learns the map from the problem to its answer:

    G : (a parameter, or a whole input function)  ->  (the solution function)

Train it once, on many solved instances. Afterwards a new instance is a forward
pass, which is microseconds.

DEEPONET, THE ARCHITECTURE USED HERE

Two networks and a dot product, which is far simpler than the name suggests:

    branch(a)  reads the problem parameter and outputs p numbers, the COEFFICIENTS
    trunk(t)   reads the query location and outputs p numbers, the BASIS FUNCTIONS
    G(a)(t) = sum over p of branch(a) * trunk(t)  +  bias

Read that as: the trunk learns a set of building-block functions of t, and the
branch decides how much of each one this particular problem needs.

That is a learned version of something old and familiar. A Fourier series writes a
function as a weighted sum of fixed sines and cosines. Here the basis is not fixed
in advance; it is learned from the family of problems you actually care about, so
far fewer terms are needed. Booklet Chapter 18.

A note on FNO, which is the other name you will meet. The Fourier Neural Operator
takes a different route to the same goal: it moves to the frequency domain with an
FFT, multiplies by learned weights there, and comes back. That makes it very
efficient on regular grids and is why it dominates in fluid dynamics benchmarks.
DeepONet is used here because it is honest to implement in numpy in forty lines
and the idea transfers.

Booklet cross-references: Chapters 18 and 19.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import sys as _sys
import matplotlib
if "ipykernel" not in _sys.modules:      # keep inline plots working in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autodiff import Node, grad, add, sub, mul, square, mean, sum_all, const
from mlp import MLP, Adam

SLATE = "#47546B"
AMBER = "#A96E08"
RED = "#9E3B2C"
GREEN = "#2E7D4F"
GREY = "#8A93A5"

T_MAX = 10.0


def exact_family(c, t, m=1.0, k=1.0, u0=1.0, v0=0.0):
    """The damped oscillator's solution, as a function of the damping c."""
    c = np.asarray(c, float)
    w = np.sqrt(k / m)
    z = c / (2.0 * np.sqrt(k * m))
    wd = w * np.sqrt(np.maximum(1 - z * z, 1e-9))
    B = (v0 + z * w * u0) / wd
    return np.exp(-z * w * t) * (u0 * np.cos(wd * t) + B * np.sin(wd * t))


class DeepONet:
    """branch(parameter) . trunk(time), which is the whole architecture."""

    def __init__(self, p=24, hidden=40, seed=0):
        self.branch = MLP([1, hidden, hidden, p], seed=seed)
        self.trunk = MLP([1, hidden, hidden, p], seed=seed + 1)
        self.bias = Node(np.zeros((1, 1)), name="b0")
        self.params = self.branch.params + self.trunk.params + [self.bias]
        self.p = p

    @property
    def n_params(self):
        return sum(q.value.size for q in self.params)

    def __call__(self, c, t):
        """c has shape (N, 1), t has shape (N, 1). Returns (N, 1)."""
        b = self.branch(const(np.asarray(c, float).reshape(-1, 1) * 2.0 - 1.0))
        tr = self.trunk(const(np.asarray(t, float).reshape(-1, 1) * 2.0 / T_MAX - 1.0))
        from autodiff import Node as _N
        prod = mul(b, tr)
        # sum along the p axis: implemented as a matmul with a column of ones
        ones = _N(np.ones((self.p, 1)))
        from autodiff import matmul
        return add(matmul(prod, ones), self.bias)

    def predict(self, c, t):
        c = np.full_like(np.asarray(t, float).reshape(-1, 1), c)
        return self(c, np.asarray(t, float).reshape(-1, 1)).value


def make_operator_data(n_params=40, n_times=40, c_range=(0.05, 0.6), seed=0):
    """Solved instances: many values of the damping, each sampled in time."""
    rng = np.random.default_rng(seed)
    cs = rng.uniform(*c_range, n_params)
    C, Tv, U = [], [], []
    for c in cs:
        t = np.sort(rng.uniform(0, T_MAX, n_times))
        C.append(np.full(n_times, c))
        Tv.append(t)
        U.append(exact_family(c, t))
    return (np.concatenate(C).reshape(-1, 1),
            np.concatenate(Tv).reshape(-1, 1),
            np.concatenate(U).reshape(-1, 1))


def fit_operator(model, C, Tv, U, epochs=3000, lr=3e-3, batch=256, seed=0):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params, lr=lr)
    n = len(C)
    for ep in range(epochs):
        idx = rng.permutation(n)[:batch]
        pred = model(C[idx], Tv[idx])
        loss = mean(square(sub(pred, const(U[idx]))))
        opt.step(grad(loss, model.params))
        if ep + 1 in (int(epochs * 0.6), int(epochs * 0.85)):
            opt.set_lr(opt.lr * 0.3)
    return model


def operator_rmse(model, cs, n=200):
    t = np.linspace(0, T_MAX, n).reshape(-1, 1)
    errs = []
    for c in cs:
        pred = model.predict(c, t)
        errs.append(np.sqrt(np.mean((pred - exact_family(c, t)) ** 2)))
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# EXPERIMENT: one training run against one training run per problem
# ---------------------------------------------------------------------------

def experiment_operator(epochs=3000, seed=0, quick=False):
    """Train the operator once. Then solve twenty new problems for free.

    The comparison being drawn is not "which is more accurate". A PINN trained on
    one instance will generally beat an operator on that instance, and it does
    here. The comparison is about what happens when you need the answer for a
    parameter you have not met, which is the situation in every design loop ever
    run.
    """
    if quick:
        epochs = 1200
    C, Tv, U = make_operator_data(seed=seed)
    net = DeepONet(seed=seed)

    t0 = time.time()
    fit_operator(net, C, Tv, U, epochs=epochs, seed=seed)
    train_time = time.time() - t0

    rng = np.random.default_rng(seed + 99)
    unseen = rng.uniform(0.08, 0.55, 20)

    t1 = time.time()
    err = operator_rmse(net, unseen)
    query_time = (time.time() - t1) / len(unseen)

    rows = [{
        "approach": "neural operator (DeepONet)",
        "trained on": "40 solved instances",
        "cost per NEW parameter": f"{query_time * 1000:.2f} ms",
        "mean RMSE over 20 unseen parameters": round(err, 4),
        "one-off training cost": f"{train_time:.0f} s",
    }]

    # What a PINN costs for the same job, measured rather than asserted.
    from pinn import DampedOscillator, Solver, fit as fit_pinn
    n_pinn = 2 if quick else 3
    pinn_errs, t2 = [], time.time()
    for c in unseen[:n_pinn]:
        sysm = DampedOscillator(c=float(c))
        s = Solver(sysm, seed=seed)
        fit_pinn(s, epochs=(1200 if quick else 3000),
                 t_col=Node(np.linspace(0, sysm.T, 200).reshape(-1, 1)), seed=seed)
        pinn_errs.append(s.rmse())
    per_pinn = (time.time() - t2) / n_pinn
    rows.append({
        "approach": "a fresh PINN for each parameter",
        "trained on": "nothing (it needs no data)",
        "cost per NEW parameter": f"{per_pinn:.0f} s",
        "mean RMSE over 20 unseen parameters": round(float(np.mean(pinn_errs)), 4),
        "one-off training cost": "none",
    })
    df = pd.DataFrame(rows)
    speedup = per_pinn / max(query_time, 1e-9)
    return df, net, unseen, speedup


def fig_operator(net, unseen):
    t = np.linspace(0, T_MAX, 300).reshape(-1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    for i, c in enumerate(sorted(unseen)[:4]):
        col = plt.cm.viridis(0.12 + 0.72 * i / 3)
        axes[0].plot(t, exact_family(c, t), lw=3.0, color=GREY,
                     label="exact" if i == 0 else None)
        axes[0].plot(t, net.predict(c, t), lw=1.9, ls="--", color=col,
                     label=f"operator, c = {c:.3f}")
    axes[0].set_xlabel("time t")
    axes[0].set_ylabel("displacement u")
    axes[0].legend(fontsize=9.5)
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Four damping values the operator never saw,\nsolved by one forward pass each",
                      fontsize=11.5, color=SLATE)

    cs = np.linspace(0.05, 0.6, 40)
    errs = [np.sqrt(np.mean((net.predict(c, t) - exact_family(c, t)) ** 2))
            for c in cs]
    axes[1].semilogy(cs, errs, lw=2.4, color=SLATE)
    axes[1].axvspan(0.05, 0.6, color=GREEN, alpha=0.07)
    axes[1].set_xlabel("damping coefficient c")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].set_title("Accuracy across the whole parameter range\n(the shaded band is where training instances came from)",
                      fontsize=11.5, color=SLATE)
    fig.suptitle("A neural operator maps a problem to its solution, so a new problem is a forward pass",
                 fontsize=12.5, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


def fig_basis(net):
    """What the trunk learned: the basis functions it decided it needed."""
    t = np.linspace(0, T_MAX, 300).reshape(-1, 1)
    tr = net.trunk(const(t * 2.0 / T_MAX - 1.0)).value
    imp = np.abs(tr).max(axis=0)
    order = np.argsort(-imp)[:6]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, j in enumerate(order):
        ax.plot(t, tr[:, j], lw=2.0, color=plt.cm.viridis(0.1 + 0.75 * i / 5),
                label=f"basis {j}")
    ax.set_xlabel("time t")
    ax.set_ylabel("trunk output")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(alpha=0.25)
    ax.set_title("The six most active learned basis functions.\n"
                 "A Fourier series would have been handed sines and cosines; these were learned from the problem family.",
                 fontsize=11.5, color=SLATE)
    fig.tight_layout()
    return fig


def main(quick=False):
    print("\nEXPERIMENT: an operator trained once, against a PINN trained per problem")
    df, net, unseen, speedup = experiment_operator(quick=quick)
    print(df.to_string(index=False))
    print(f"\n  Once trained, the operator answers a new problem about "
          f"{speedup:,.0f} times faster.")
    print("  It is less accurate on any single problem, which is the trade being made.")
    print("  For a design loop over thousands of parameter values, that trade is")
    print("  usually the difference between feasible and impossible.")
    return df, net, unseen


if __name__ == "__main__":
    import sys as _s
    main(quick="--quick" in _s.argv)
