"""
gradient_descent.py, the Gradient Descent Explorer.

The Evolution of Machine Learning and AI · Week 3 Lab
South East Technological University

Everything in this module that learns, learns by this loop. Before you meet it
buried inside PyTorch, meet it here in twelve lines of numpy, where you can see
every moving part and break each one on purpose.

Booklet cross-references:
    Chapter 8   the hillwalker in fog
    Chapter 9   learning rate, convergence, divergence
    Chapter 10  local minima, saddle points, momentum

The four experiments this file supports, which are the four things the lab asks
you to see with your own eyes:

    1. the learning rate is the whole ballgame       lr_sweep()
    2. a loss surface is not a bowl                  landscape_2d()
    3. local minima are a real phenomenon            local_minima_demo()
    4. momentum is not magic, it is memory           compare_optimisers()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sys
import matplotlib
if "ipykernel" not in sys.modules:      # keep inline plots working in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SLATE, AMBER, RED, BLUE, GREY, GREEN = "#47546B", "#A96E08", "#9E3B2C", "#4E7291", "#8A93A5", "#2E7D4F"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
})


# ---------------------------------------------------------------------------
# The algorithm. This is all of it.
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """Everything that happened during one descent, so you can plot it."""
    xs: np.ndarray          # the parameter at each step
    losses: np.ndarray      # the loss at each step
    grads: np.ndarray       # the gradient at each step
    lr: float
    diverged: bool


def descend(f, df, x0, lr=0.1, steps=60, momentum=0.0, clip=1e6) -> Run:
    """Gradient descent, written out.

        x <- x - lr * df(x)

    That single line is the entire method, and every optimiser you meet later
    (momentum, RMSProp, Adam) is a variation on how the step is scaled or
    remembered. `momentum` keeps a running velocity, which is the smallest
    useful variation and the one that explains the rest.
    """
    x = float(x0)
    v = 0.0
    xs, losses, grads = [x], [float(f(x))], []
    diverged = False

    for _ in range(steps):
        g = float(df(x))
        grads.append(g)
        v = momentum * v - lr * g          # velocity: memory of past gradients
        x = x + v
        if not np.isfinite(x) or abs(x) > clip:
            diverged = True
            break
        xs.append(x)
        losses.append(float(f(x)))

    return Run(np.array(xs), np.array(losses), np.array(grads), lr, diverged)


# ---------------------------------------------------------------------------
# Test surfaces
# ---------------------------------------------------------------------------

def bowl(x):        return 0.5 * x ** 2
def bowl_d(x):      return x

def wiggly(x):
    """A convex bowl with a dent in it: one global minimum, one local."""
    return 0.06 * x ** 4 - 0.4 * x ** 3 + 0.2 * x ** 2 + 2.0 * x + 8.0

def wiggly_d(x):
    return 0.24 * x ** 3 - 1.2 * x ** 2 + 0.4 * x + 2.0

def rosen(x, y):
    """Rosenbrock: the classic banana valley. Easy to enter, hard to follow."""
    return (1 - x) ** 2 + 20 * (y - x ** 2) ** 2

def rosen_grad(x, y):
    return (-2 * (1 - x) - 80 * x * (y - x ** 2), 40 * (y - x ** 2))


# ---------------------------------------------------------------------------
# Experiment 1: the learning rate is the whole ballgame
# ---------------------------------------------------------------------------

def lr_sweep(rates=(0.02, 0.1, 0.9, 1.98, 2.05), x0=4.0, steps=40):
    """Same surface, same start, five step sizes. Returns (fig, table)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    xs = np.linspace(-5, 5, 400)
    axes[0].plot(xs, bowl(xs), color=GREY, lw=1.4)

    rows = []
    colours = [BLUE, GREEN, AMBER, RED, "#5B2C6F"]
    for lr, c in zip(rates, colours):
        r = descend(bowl, bowl_d, x0, lr=lr, steps=steps)
        axes[0].plot(r.xs, bowl(r.xs), "o-", ms=3, lw=1.2, color=c, alpha=0.85,
                     label=f"lr = {lr}")
        axes[1].plot(r.losses, lw=1.6, color=c, label=f"lr = {lr}")
        rows.append({
            "lr": lr,
            "steps_taken": len(r.xs) - 1,
            "final_x": float(r.xs[-1]) if not r.diverged else float("nan"),
            "final_loss": float(r.losses[-1]) if not r.diverged else float("inf"),
            "diverged": r.diverged,
            "verdict": _verdict(r, lr),
        })

    axes[0].set_title("The path taken")
    axes[0].set_xlabel("parameter"); axes[0].set_ylabel("loss")
    axes[0].set_ylim(-1, 14); axes[0].set_xlim(-5, 5)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_yscale("symlog")
    axes[1].set_title("Loss against step")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("loss (symlog)")
    axes[1].legend(frameon=False, fontsize=8)
    plt.tight_layout()

    import pandas as pd
    return fig, pd.DataFrame(rows)


def _verdict(r: Run, lr: float) -> str:
    """Name the failure mode. There are only four, and you will meet all of them."""
    oscillating = len(r.xs) > 2 and np.any(np.diff(np.sign(r.xs)) != 0)
    if r.diverged or r.losses[-1] > r.losses[0]:
        return "DIVERGING: every step overshoots further than the last"
    if r.losses[-1] > 0.01:
        return ("crawling, and oscillating" if oscillating
                else "crawling: still descending when the budget ran out")
    return ("converged, oscillating across the minimum" if oscillating
            else "converged smoothly")


# ---------------------------------------------------------------------------
# Experiment 2: a loss surface is not a bowl
# ---------------------------------------------------------------------------

def descend_2d(f, grad, p0, lr=0.002, steps=400, momentum=0.0):
    x, y = p0
    vx = vy = 0.0
    path = [(x, y)]
    for _ in range(steps):
        gx, gy = grad(x, y)
        vx = momentum * vx - lr * gx
        vy = momentum * vy - lr * gy
        x, y = x + vx, y + vy
        if not (np.isfinite(x) and np.isfinite(y)) or abs(x) > 1e3:
            break
        path.append((x, y))
    return np.array(path)


def landscape_2d(lr=0.002, steps=600):
    """Contours of the Rosenbrock valley, with and without momentum."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    X, Y = np.meshgrid(np.linspace(-1.6, 1.6, 300), np.linspace(-0.6, 2.2, 300))
    Z = rosen(X, Y)
    ax.contour(X, Y, Z, levels=np.logspace(-0.6, 3.2, 22), colors=GREY,
               linewidths=0.5, alpha=0.7)

    for mom, c, lab in ((0.0, RED, "plain gradient descent"),
                        (0.9, GREEN, "with momentum 0.9")):
        p = descend_2d(rosen, rosen_grad, (-1.2, 1.4), lr=lr, steps=steps, momentum=mom)
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.6, color=c, alpha=0.9, label=lab)
        ax.plot(p[-1, 0], p[-1, 1], "o", ms=6, color=c)

    ax.plot(1, 1, "*", ms=14, color=AMBER, label="the minimum")
    ax.set_title("Same surface, same start, same learning rate")
    ax.set_xlabel("parameter 1"); ax.set_ylabel("parameter 2")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Experiment 3: local minima are real
# ---------------------------------------------------------------------------

def local_minima_demo(starts=(-2.5, 1.0, 5.5), lr=0.02, steps=200):
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    xs = np.linspace(-3.5, 7.5, 500)
    ax.plot(xs, wiggly(xs), color=GREY, lw=1.6)

    rows = []
    for x0, c in zip(starts, (BLUE, AMBER, GREEN)):
        r = descend(wiggly, wiggly_d, x0, lr=lr, steps=steps)
        ax.plot(r.xs, wiggly(r.xs), "o-", ms=2.5, lw=1.1, color=c, alpha=0.85)
        ax.plot(r.xs[0], wiggly(r.xs[0]), "o", ms=8, color=c)
        ax.annotate(f"start {x0}", (r.xs[0], wiggly(r.xs[0])),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, color=c, fontweight="bold")
        rows.append({"start": x0, "final_x": round(float(r.xs[-1]), 3),
                     "final_loss": round(float(r.losses[-1]), 4)})

    ax.set_title("Identical algorithm, identical learning rate, different starting point")
    ax.set_xlabel("parameter"); ax.set_ylabel("loss")
    plt.tight_layout()
    import pandas as pd
    return fig, pd.DataFrame(rows)


def compare_optimisers(x0=4.0, steps=60):
    """Momentum against plain descent on the same surface."""
    import pandas as pd
    rows = []
    for mom in (0.0, 0.5, 0.9):
        r = descend(bowl, bowl_d, x0, lr=0.1, steps=steps, momentum=mom)
        first = int(np.argmax(r.losses < 0.01)) if np.any(r.losses < 0.01) else -1
        rows.append({"momentum": mom,
                     "steps_to_loss<0.01": first if first > 0 else "not reached",
                     "best_loss": round(float(r.losses.min()), 6),
                     "final_loss": round(float(r.losses[-1]), 6)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def main(outdir="figures"):
    import pandas as pd
    out = Path(outdir); out.mkdir(exist_ok=True)
    pd.set_option("display.width", 170)

    fig, table = lr_sweep()
    fig.savefig(out / "01_lr_sweep.png", bbox_inches="tight")
    print("EXPERIMENT 1 · the learning rate is the whole ballgame\n")
    print(table.to_string(index=False), "\n")

    landscape_2d().savefig(out / "02_landscape.png", bbox_inches="tight")

    fig, table = local_minima_demo()
    fig.savefig(out / "03_local_minima.png", bbox_inches="tight")
    print("EXPERIMENT 3 · three starting points, one algorithm\n")
    print(table.to_string(index=False), "\n")

    print("EXPERIMENT 4 · momentum is memory\n")
    print(compare_optimisers().to_string(index=False))
    print(f"\nFigures written to {out.resolve()}")


if __name__ == "__main__":
    main()
