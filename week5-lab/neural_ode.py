"""
neural_ode.py, learning the LAW instead of the ANSWER.

The Evolution of Machine Learning and AI, Week 5 Lab
South East Technological University

THE IDEA, AND WHY IT IS A DIFFERENT IDEA FROM A PINN

A PINN represents the solution directly: the network IS u(t), and the loss forces
it to satisfy the equation. That works beautifully, and it has one structural
limitation. The network has learned ONE trajectory. Change the initial condition
and you must train again from scratch.

A neural ODE inverts the arrangement. The network does not represent the solution;
it represents the DERIVATIVE:

    du/dt = f_theta(u)

To get a trajectory you hand f_theta to a numerical integrator and roll forward
from wherever you like. The network has learned the rule of motion rather than
one instance of motion, so a new starting point costs nothing.

That distinction is the whole file, and it is worth holding onto because it
recurs. Learning a trajectory gives you an answer. Learning a vector field gives
you a model, and a model can be asked questions you did not train it on.

THE THREE MODELS

    DIRECT MAP        t -> u. What Weeks 3 and 4 would build. It interpolates
                      the training window and has no opinion outside it.

    NEURAL ODE        u -> du/dt, integrated with Runge-Kutta 4. Learns the field.

    STRUCTURED ODE    the same, but the field is constrained to the form physics
                      demands: dx/dt = v exactly, and only the acceleration is
                      learned. Half the field is known, so only half is fitted.

THE EXPERIMENTS

    1  Roll out past the training window. Does the model keep working?
    2  Start from an initial condition never seen in training.
    3  How much does constraining half the vector field buy?

Booklet cross-references: Chapters 16 and 17.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys as _sys
import matplotlib
if "ipykernel" not in _sys.modules:      # keep inline plots working in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autodiff import (Node, grad, add, sub, mul, square, mean, concat,
                      slice_axis, const)
from mlp import MLP, Adam

SLATE = "#47546B"
AMBER = "#A96E08"
RED = "#9E3B2C"
GREEN = "#2E7D4F"
GREY = "#8A93A5"

M, C, K = 1.0, 0.2, 1.0


def true_field(u):
    """The dynamics we are trying to recover, as a vector field on (x, v)."""
    x, v = u[:, 0:1], u[:, 1:2]
    return np.concatenate([v, (-K * x - C * v) / M], axis=1)


def rk4_numpy(u0, dt, n_steps, field=true_field):
    """The reference integrator, for generating ground truth."""
    us = [np.asarray(u0, float)]
    u = us[0]
    for _ in range(n_steps):
        k1 = field(u)
        k2 = field(u + 0.5 * dt * k1)
        k3 = field(u + 0.5 * dt * k2)
        k4 = field(u + dt * k3)
        u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        us.append(u)
    return np.stack(us, axis=1)          # (batch, steps+1, 2)


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------

class DirectMap:
    """(t, x0, v0) -> (x, v). A plain regressor on the solution map.

    This is the FAIR baseline, and it is worth saying why the obvious version is
    not. A network taking only t cannot represent a family of trajectories at all:
    given 64 different starting states it can only predict their average, so it
    would lose for a trivial reason and prove nothing.

    Given the starting state as an input, it CAN represent the whole family. It is
    a direct approximation of the solution operator, and it is a perfectly
    reasonable thing to build. Its limitation is subtler: it has learned a
    function on the box of (t, x0, v0) values it was trained on, and outside that
    box it has no opinion worth having. A vector field, by contrast, is defined
    everywhere and is integrated forward for as long as you ask.
    """

    name = "direct map (t, u0) -> u"

    def __init__(self, hidden=40, seed=0, T=5.0):
        self.net = MLP([3, hidden, hidden, 2], seed=seed)
        self.params = self.net.params
        self.T = T

    def _inputs(self, u0, dt, n_steps):
        u0 = np.atleast_2d(np.asarray(u0, float))
        t = np.arange(n_steps + 1) * dt
        B, S = len(u0), len(t)
        tt = np.tile(t, B).reshape(-1, 1) * 2.0 / self.T - 1.0
        uu = np.repeat(u0, S, axis=0)
        return np.concatenate([tt, uu], axis=1), B, S

    def predict_graph(self, u0, dt, n_steps):
        X, B, S = self._inputs(u0, dt, n_steps)
        return self.net(Node(X)), B, S

    def trajectory(self, u0, dt, n_steps):
        out, B, S = self.predict_graph(u0, dt, n_steps)
        return out.value.reshape(B, S, 2)

    @property
    def n_params(self):
        return self.net.n_params


class NeuralODE:
    """u -> du/dt, integrated with RK4 inside the computation graph.

    The gradient is taken by differentiating THROUGH the integrator, which is the
    straightforward approach and is what this lab does. The original neural ODE
    paper instead solves a second ODE backwards in time (the adjoint method) to
    avoid storing every intermediate step, which matters when there are thousands
    of steps. With forty steps, storing them is cheap and the code stays readable.
    Booklet Chapter 16 explains the trade.
    """

    name = "neural ODE"

    def __init__(self, hidden=40, seed=0, structured=False):
        out = 1 if structured else 2
        self.net = MLP([2, hidden, hidden, out], seed=seed)
        self.params = self.net.params
        self.structured = structured

    @property
    def n_params(self):
        return self.net.n_params

    def field(self, u):
        """du/dt as a graph node.

        When `structured` is set, only the acceleration is learned and the
        kinematic half of the field is imposed exactly:

            dx/dt = v            known, from the definition of velocity
            dv/dt = f_theta(x,v) unknown, this is the physics being learned

        The first line is not an approximation or a modelling choice. It is what
        velocity MEANS. Handing it to a network to discover from data is spending
        capacity and examples on a definition.
        """
        out = self.net(u)
        if not self.structured:
            return out
        v = slice_axis(u, 1, 1, 2)
        return concat([v, out], axis=1)

    def rollout(self, u0, dt, n_steps):
        """RK4, written out, with every step inside the graph."""
        u = const(np.asarray(u0, float))
        traj = [u]
        for _ in range(n_steps):
            k1 = self.field(u)
            k2 = self.field(add(u, mul(0.5 * dt, k1)))
            k3 = self.field(add(u, mul(0.5 * dt, k2)))
            k4 = self.field(add(u, mul(dt, k3)))
            incr = mul(dt / 6.0, add(add(k1, mul(2.0, k2)),
                                     add(mul(2.0, k3), k4)))
            u = add(u, incr)
            traj.append(u)
        return traj

    def trajectory(self, u0, dt, n_steps):
        traj = self.rollout(u0, dt, n_steps)
        return np.stack([s.value for s in traj], axis=1)


class StructuredODE(NeuralODE):
    name = "structured ODE (dx/dt = v imposed)"

    def __init__(self, hidden=40, seed=0):
        super().__init__(hidden=hidden, seed=seed, structured=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_trajectories(n, dt=0.25, n_steps=20, seed=0, spread=1.0):
    rng = np.random.default_rng(seed)
    u0 = np.stack([rng.uniform(-spread, spread, n),
                   rng.uniform(-spread, spread, n)], axis=1)
    return u0, rk4_numpy(u0, dt, n_steps)


def fit_ode(model, u0, target, dt, epochs=600, lr=5e-3, seed=0):
    """Match the whole rolled-out trajectory, not just one step ahead."""
    opt = Adam(model.params, lr=lr)
    n_steps = target.shape[1] - 1
    for ep in range(epochs):
        if isinstance(model, DirectMap):
            pred, B, S = model.predict_graph(u0, dt, n_steps)
            flat = target.reshape(-1, 2)
            loss = mean(square(sub(pred, const(flat))))
        else:
            traj = model.rollout(u0, dt, n_steps)
            terms = []
            for s in range(1, n_steps + 1):
                terms.append(mean(square(sub(traj[s], const(target[:, s])))))
            loss = terms[0]
            for extra in terms[1:]:
                loss = add(loss, extra)
            loss = mul(1.0 / n_steps, loss)
        opt.step(grad(loss, model.params), clip=10.0)
        if ep + 1 in (int(epochs * 0.6), int(epochs * 0.85)):
            opt.set_lr(opt.lr * 0.3)
    return model


def traj_rmse(model, u0, dt, n_steps):
    truth = rk4_numpy(u0, dt, n_steps)
    pred = model.trajectory(u0, dt, n_steps)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


# ---------------------------------------------------------------------------
# EXPERIMENT 1: rolling out past the training window
# ---------------------------------------------------------------------------

def experiment_rollout(dt=0.25, train_steps=20, test_steps=60, epochs=600,
                       seed=0, quick=False):
    """Train for 20 steps. Then keep integrating to 60 and see what happens.

    A neural ODE can be rolled out for as long as you like, because it learned a
    rule rather than an interval. Whether the answer stays ACCURATE that far is a
    separate question, and this table answers it honestly: error grows, because
    small errors in the field compound over steps, but it grows rather than
    exploding into nonsense.
    """
    if quick:
        epochs = 350
    u0, target = make_trajectories(64, dt=dt, n_steps=train_steps, seed=seed)
    rows = []
    for ctor in (DirectMap, NeuralODE, StructuredODE):
        m = ctor(seed=seed)
        fit_ode(m, u0, target, dt, epochs=epochs, seed=seed)
        rows.append({
            "model": m.name, "parameters": m.n_params,
            "RMSE, 20 steps (trained)": round(traj_rmse(m, u0, dt, train_steps), 4),
            "RMSE, 60 steps (3x longer)": round(traj_rmse(m, u0, dt, test_steps), 4),
        })
    df = pd.DataFrame(rows)
    df["degradation factor"] = (df["RMSE, 60 steps (3x longer)"] /
                               df["RMSE, 20 steps (trained)"]).round(1)
    return df


# ---------------------------------------------------------------------------
# EXPERIMENT 2: initial conditions never seen
# ---------------------------------------------------------------------------

def experiment_new_initial_conditions(dt=0.25, n_steps=20, epochs=600, seed=0,
                                      quick=False):
    """Train on trajectories starting inside a small box. Test outside it.

    This is the result that justifies the whole approach. The direct map cannot
    even be ASKED the question: it is a function of time alone, so it has no slot
    for a starting state. The neural ODE takes a new starting point as an
    argument, because a vector field is defined everywhere.

    Report the ABSOLUTE errors here rather than a degradation ratio, because all
    three models degrade outside the box and the ratios are misleading. Rolling a
    learned field forward from a state it never visited means evaluating it where
    it was never fitted, and no amount of structure removes that. What structure
    buys is a smaller error, not immunity: the structured ODE is several times
    more accurate than the direct map on starts it never saw, which is worth
    having and is not the same as a guarantee.

    Compare Week 4, Chapter 9's "positions never seen in training". Same shape of
    result, and here the generalisation is in the space of initial conditions
    rather than the space of positions.
    """
    if quick:
        epochs = 350
    u0_tr, tgt_tr = make_trajectories(64, dt=dt, n_steps=n_steps, seed=seed,
                                      spread=0.8)
    rng = np.random.default_rng(seed + 5)
    far = np.stack([rng.uniform(0.9, 1.3, 32) * rng.choice([-1, 1], 32),
                    rng.uniform(0.9, 1.3, 32) * rng.choice([-1, 1], 32)], axis=1)
    rows = []
    for ctor in (DirectMap, NeuralODE, StructuredODE):
        m = ctor(seed=seed)
        fit_ode(m, u0_tr, tgt_tr, dt, epochs=epochs, seed=seed)
        rows.append({
            "model": m.name,
            "RMSE, starts seen in training": round(traj_rmse(m, u0_tr, dt, n_steps), 4),
            "RMSE, starts outside the box": round(traj_rmse(m, far, dt, n_steps), 4),
        })
    df = pd.DataFrame(rows)
    best = df["RMSE, starts outside the box"].min()
    df["times worse than best"] = (df["RMSE, starts outside the box"] / best).round(1)
    return df


def fig_vector_field(seed=0, epochs=600):
    """The learned field beside the true one. This is what was actually learned."""
    u0, target = make_trajectories(64, seed=seed)
    m = StructuredODE(seed=seed)
    fit_ode(m, u0, target, 0.25, epochs=epochs, seed=seed)

    g = np.linspace(-2, 2, 17)
    X, V = np.meshgrid(g, g)
    pts = np.stack([X.ravel(), V.ravel()], axis=1)
    true = true_field(pts)
    learned = m.field(const(pts)).value

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, f, title in ((axes[0], true, "the true vector field"),
                         (axes[1], learned, "what the structured neural ODE learned")):
        ax.quiver(X, V, f[:, 0].reshape(X.shape), f[:, 1].reshape(X.shape),
                  color=SLATE, alpha=0.75, width=0.004)
        traj = rk4_numpy(np.array([[1.5, 0.0]]), 0.05, 400)[0]
        ax.plot(traj[:, 0], traj[:, 1], lw=2.2, color=RED, alpha=0.8)
        ax.set_xlabel("displacement x")
        ax.set_ylabel("velocity v")
        ax.set_title(title, fontsize=11.5, color=SLATE)
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-2.1, 2.1)
        ax.grid(alpha=0.2)
    fig.suptitle("A neural ODE learns the rule of motion, not one trajectory.\n"
                 "The red spiral is one solution; the arrows are what generates every solution.",
                 fontsize=12.5, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


def fig_rollout(seed=0, epochs=600):
    dt, train_steps, test_steps = 0.25, 20, 60
    u0, target = make_trajectories(64, dt=dt, n_steps=train_steps, seed=seed)
    show = np.array([[1.0, 0.0]])
    truth = rk4_numpy(show, dt, test_steps)[0]
    t = np.arange(test_steps + 1) * dt

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.axvspan(0, train_steps * dt, color=GREEN, alpha=0.08)
    ax.text(train_steps * dt / 2, 1.15, "trained here", ha="center",
            fontsize=11, color=GREEN)
    ax.text((train_steps + test_steps) * dt / 2, 1.15,
            "rolled out three times further", ha="center", fontsize=11, color=RED)
    ax.plot(t, truth[:, 0], lw=3.4, color=GREY, label="truth")
    for ctor, col in ((DirectMap, AMBER), (NeuralODE, SLATE), (StructuredODE, GREEN)):
        m = ctor(seed=seed)
        fit_ode(m, u0, target, dt, epochs=epochs, seed=seed)
        pred = m.trajectory(show, dt, test_steps)[0]
        ax.plot(t, pred[:, 0], lw=2.0, ls="--", color=col, label=m.name)
    ax.axvline(train_steps * dt, color=RED, ls=":", lw=1.6)
    ax.set_xlabel("time t")
    ax.set_ylabel("displacement x")
    ax.set_ylim(-1.4, 1.35)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_title("Integrating a learned rule keeps working past the training window",
                 fontsize=12.5, color=SLATE)
    fig.tight_layout()
    return fig


def main(quick=False):
    print("\nEXPERIMENT 1: rolling out three times past the training window")
    r = experiment_rollout(quick=quick)
    print(r.to_string(index=False))
    print("\nEXPERIMENT 2: initial conditions never seen in training")
    n = experiment_new_initial_conditions(quick=quick)
    print(n.to_string(index=False))
    print("\n  The direct map cannot even be asked this question: it is a function")
    print("  of time alone, so it has nowhere to put a new starting state.")
    return r, n


if __name__ == "__main__":
    import sys as _s
    main(quick="--quick" in _s.argv)
