"""
sequences.py, recurrence as an assumption, and the price it charges.

The Evolution of Machine Learning and AI, Week 4 Lab
South East Technological University

A convolution assumes that meaning is LOCAL and that it is the SAME
everywhere. A recurrent network assumes something different:

    ORDER MATTERS         the sequence is not a bag of items; position
                          in time carries meaning.
    THE PAST SUMMARISES   everything relevant about steps 1 to t can be
                          squeezed into one fixed-size state vector h_t.
    STATIONARITY IN TIME  the same update rule applies at every step, so
                          one set of weights is reused for all of them.

The third assumption is the same trick as a convolution, rotated ninety
degrees: a CNN shares weights across space, an RNN shares them across time.
That is why an RNN can read a sentence of any length with a fixed parameter
count, exactly as a CNN can scan an image of any size.

The second assumption is the one that hurts. This file measures the damage.

    EXPERIMENT 1  The vanishing gradient through time, measured. How much
                  does the loss at the end of a sequence care about the input
                  at the start? Answer: less and less, exponentially, and the
                  exponent is the sequence length.

    EXPERIMENT 2  A memory task. Remember the first symbol, ignore T steps of
                  distractors, report it at the end. The vanilla RNN fails as
                  T grows. The LSTM does not, and the reason is visible in
                  three lines of its forward pass.

Booklet cross-references:
    Chapter 12  sequences, and why order is a different kind of structure
    Chapter 13  the RNN, recurrence as weight sharing through time
    Chapter 14  backpropagation through time and the vanishing gradient
    Chapter 15  LSTMs and GRUs, gates as learned memory control
    Chapter 16  where recurrence fails

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

from optim import AdamMixin

SLATE = "#47546B"
AMBER = "#A96E08"
RED = "#9E3B2C"
GREEN = "#2E7D4F"
BLUE = "#4E7291"
GREY = "#8A93A5"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# EXPERIMENT 1: the vanishing gradient through time, measured
# ---------------------------------------------------------------------------

def gradient_through_time(T=60, hidden=32, scales=(0.5, 0.9, 1.0, 1.2), seed=0):
    """How much does step T's hidden state depend on step 1's input?

    Unroll a recurrent network for T steps and ask for the size of

        d h_T / d h_1

    The chain rule turns that into a product of T Jacobians, one per step, and
    every one of them contains the same recurrent weight matrix W. So the
    quantity behaves like W raised to the power T.

    That is the whole problem, and it is arithmetic, not a bug. If the relevant
    scale of W is a little below 1, the product decays exponentially and step 1
    becomes invisible from step T. If it is a little above 1, the product
    explodes and training blows up. The safe setting is exactly on the knife
    edge, which is not a setting you can reliably arrange.

    This is Week 3's vanishing gradient, Chapter 15, with one crucial change.
    In a deep feedforward network the layers have DIFFERENT weight matrices, so
    the factors are unrelated and the decay is partly luck. Here it is the SAME
    matrix every time, so the decay is systematic. Depth in time is worse than
    depth in layers.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in scales:
        W = rng.normal(0, s / np.sqrt(hidden), (hidden, hidden))
        h = rng.normal(0, 0.5, hidden)
        J = np.eye(hidden)
        norms = []
        for t in range(T):
            pre = W @ h
            h = np.tanh(pre)
            D = np.diag(1.0 - h ** 2)          # derivative of tanh
            J = D @ W @ J                       # accumulate d h_t / d h_0
            norms.append(np.linalg.norm(J, 2))
        rows.append({"W scale": s, "norms": np.array(norms)})

    tbl = pd.DataFrame([{
        "recurrent weight scale": r["W scale"],
        "after 5 steps": f"{r['norms'][4]:.3e}",
        "after 20 steps": f"{r['norms'][19]:.3e}",
        "after 60 steps": f"{r['norms'][59]:.3e}",
        "verdict": ("vanishes" if r["norms"][59] < 1e-4
                    else "explodes" if r["norms"][59] > 1e3
                    else "usable"),
    } for r in rows])
    return tbl, rows


def fig_vanishing(rows, T=60):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    cols = [BLUE, GREEN, SLATE, RED]
    for r, c in zip(rows, cols):
        ax.semilogy(range(1, T + 1), np.maximum(r["norms"], 1e-30),
                    lw=2.4, color=c, label=f"scale {r['W scale']}")
    ax.axhline(1.0, color=GREY, ls="--", lw=1.2)
    ax.axhspan(1e-30, 1e-4, color=RED, alpha=0.07)
    ax.text(2, 1e-8, "below here, step 1 is invisible from step T",
            fontsize=9.5, color=RED)
    ax.set_xlabel("steps back through time")
    ax.set_ylabel(r"size of $\partial h_T / \partial h_1$  (log scale)")
    ax.set_title("The same matrix, multiplied by itself, T times.\n"
                 "This is not a bug in anyone's code. It is what a product does.",
                 fontsize=12, color=SLATE, weight="bold")
    ax.legend(fontsize=10, title="recurrent weight scale")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# The task: remember the first symbol
# ---------------------------------------------------------------------------

def make_memory_task(n, T, n_symbols=4, seed=0):
    """Step 1 carries a symbol. Steps 2..T carry noise. At step T, report it.

    Deliberately the simplest possible long-range dependency: exactly one bit
    of information has to survive exactly T steps, and nothing else matters at
    all. If a model cannot do this, it certainly cannot track the subject of a
    sentence across a paragraph.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_symbols, n)
    X = np.zeros((n, T, n_symbols + 1))
    X[np.arange(n), 0, y] = 1.0                 # the symbol, at step 1
    X[:, 1:, n_symbols] = 1.0                   # a "nothing to see here" marker
    X[:, 1:, :n_symbols] = rng.normal(0, 0.1, (n, T - 1, n_symbols))
    return X, y


# ---------------------------------------------------------------------------
# A vanilla RNN, with backpropagation through time written out
# ---------------------------------------------------------------------------

class VanillaRNN(AdamMixin):
    """h_t = tanh(x_t Wx + h_(t-1) Wh + b);  predict from h_T only.

    Note there is exactly one Wh, used at every step. That is weight sharing
    across time, and it is the reason this model has a fixed size no matter
    how long the sequence is. It is also, unavoidably, the reason the gradient
    is a power of one matrix.
    """

    def __init__(self, n_in, hidden, n_out, seed=0, scale=1.0):
        rng = np.random.default_rng(seed)
        self.Wx = rng.normal(0, scale / np.sqrt(n_in), (n_in, hidden))
        self.Wh = rng.normal(0, scale / np.sqrt(hidden), (hidden, hidden))
        self.b = np.zeros(hidden)
        self.Wo = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, n_out))
        self.bo = np.zeros(n_out)
        self.hidden = hidden
        self.params = ["Wx", "Wh", "b", "Wo", "bo"]

    @property
    def n_params(self):
        return sum(getattr(self, p).size for p in self.params)

    def forward(self, X):
        N, T, _ = X.shape
        H = np.zeros((N, T + 1, self.hidden))
        for t in range(T):
            H[:, t + 1] = np.tanh(X[:, t] @ self.Wx + H[:, t] @ self.Wh + self.b)
        self.X, self.H = X, H
        return H[:, -1] @ self.Wo + self.bo

    def backward(self, d_logits):
        """Backpropagation through time: the same chain rule, T steps of it."""
        X, H = self.X, self.H
        N, T, _ = X.shape
        self.dWo = H[:, -1].T @ d_logits
        self.dbo = d_logits.sum(0)
        dh = d_logits @ self.Wo.T

        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)
        first_step_norm = None
        for t in reversed(range(T)):
            dpre = dh * (1.0 - H[:, t + 1] ** 2)
            self.dWx += X[:, t].T @ dpre
            self.dWh += H[:, t].T @ dpre
            self.db += dpre.sum(0)
            dh = dpre @ self.Wh.T          # <-- the same Wh, every single step
            if t == 0:
                first_step_norm = float(np.linalg.norm(dpre) / N)
        self.first_step_grad = first_step_norm



# ---------------------------------------------------------------------------
# An LSTM, with the gates written out
# ---------------------------------------------------------------------------

class LSTM(AdamMixin):
    """The fix, and it is worth understanding why it is a fix.

        f = sigmoid(...)      forget gate: how much of the old cell to keep
        i = sigmoid(...)      input gate: how much of the new candidate to add
        o = sigmoid(...)      output gate: how much of the cell to reveal
        g = tanh(...)         the candidate value itself

        c_t = f * c_(t-1) + i * g            the cell, updated ADDITIVELY
        h_t = o * tanh(c_t)                  the visible state

    Look at the cell line. The old cell is MULTIPLIED BY f AND ADDED TO, not
    passed through a weight matrix and a squashing function. So the gradient
    flowing back along the cell is multiplied by f at each step, and f is
    something the network CHOOSES. If it learns f close to 1, the gradient
    passes through almost unchanged for hundreds of steps.

    That is the entire idea. An RNN's memory decays at a rate fixed by its
    weights; an LSTM's memory decays at a rate it controls per step and per
    dimension. Gates are not extra capacity. They are learned control over how
    fast the past is forgotten.
    """

    def __init__(self, n_in, hidden, n_out, seed=0, forget_bias=3.0):
        rng = np.random.default_rng(seed)
        H = hidden
        self.W = rng.normal(0, 1.0 / np.sqrt(n_in + H), (n_in + H, 4 * H))
        self.b = np.zeros(4 * H)
        # Forget-gate bias starts POSITIVE, a standard and important trick:
        # begin by remembering nearly everything, and learn what to discard.
        # The value matters more than most people expect. sigmoid(1.5) = 0.82,
        # which halves a memory every 3.5 steps, so an LSTM initialised that way
        # starts out unable to see 90 steps back and has to learn its way out.
        # sigmoid(3.0) = 0.953 halves every 14 steps. The lab notes record what
        # happens at 1.5: the LSTM scored WORSE than a plain RNN on the
        # length-90 task, which looked like an architectural result and was
        # really a bad initialisation.
        self.b[:H] = forget_bias
        self.Wo = rng.normal(0, 1.0 / np.sqrt(H), (H, n_out))
        self.bo = np.zeros(n_out)
        self.H = H
        self.params = ["W", "b", "Wo", "bo"]

    @property
    def n_params(self):
        return sum(getattr(self, p).size for p in self.params)

    def forward(self, X):
        N, T, _ = X.shape
        H = self.H
        h = np.zeros((N, H))
        c = np.zeros((N, H))
        self.cache = []
        for t in range(T):
            z = np.concatenate([X[:, t], h], axis=1)
            a = z @ self.W + self.b
            f = sigmoid(a[:, :H])
            i = sigmoid(a[:, H:2 * H])
            o = sigmoid(a[:, 2 * H:3 * H])
            g = np.tanh(a[:, 3 * H:])
            c_prev = c
            c = f * c_prev + i * g
            tc = np.tanh(c)
            h = o * tc
            self.cache.append((z, f, i, o, g, c_prev, c, tc, h))
        self.h_last = h
        return h @ self.Wo + self.bo

    def backward(self, d_logits):
        H = self.H
        self.dWo = self.h_last.T @ d_logits
        self.dbo = d_logits.sum(0)
        dh = d_logits @ self.Wo.T
        dc = np.zeros_like(dh)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        first = None
        for t in reversed(range(len(self.cache))):
            z, f, i, o, g, c_prev, c, tc, h = self.cache[t]
            do = dh * tc
            dc = dc + dh * o * (1.0 - tc ** 2)
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            da = np.concatenate([
                df * f * (1 - f),
                di * i * (1 - i),
                do * o * (1 - o),
                dg * (1 - g ** 2),
            ], axis=1)
            self.dW += z.T @ da
            self.db += da.sum(0)
            dz = da @ self.W.T
            dh = dz[:, -H:]
            dc = dc * f                 # <-- the gradient highway: multiply by f
            if t == 0:
                first = float(np.linalg.norm(da) / len(z))
        self.first_step_grad = first



# ---------------------------------------------------------------------------
# Training and EXPERIMENT 2
# ---------------------------------------------------------------------------

def ce_loss(logits, y):
    p = softmax(logits)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p.copy()
    d[np.arange(n), y] -= 1.0
    return loss, d / n


def fit_seq(model, X, y, epochs=60, lr=0.1, batch=32, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    hist, first_grads = [], []
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            logits = model.forward(X[b])
            loss, d = ce_loss(logits, y[b])
            model.backward(d)
            model.step(lr)
            tot += loss * len(b)
        hist.append(tot / n)
        first_grads.append(model.first_step_grad)
    return hist, first_grads


def score_seq(model, X, y, batch=128):
    preds = []
    for s in range(0, len(X), batch):
        preds.append(model.forward(X[s:s + batch]).argmax(1))
    return float((np.concatenate(preds) == y).mean())


def experiment_memory(lengths=(10, 40, 90, 160), hidden=24, n_train=400,
                      epochs=100, seed=0, quick=False):
    """Remember the first symbol across T steps. Chance is 25%.

    Both models see the same data, train for the same number of epochs, and use
    the same optimiser. The only difference is whether the path from step 1 to
    step T is a repeated matrix multiply or a gated addition.

    Read the gradient columns alongside the accuracy columns, because together
    they say something more careful than "LSTMs are better".

    At T = 90 the RNN's gradient at step 1 is about 1e-18 and it scores near
    chance; the LSTM's is about 1e-05 and it scores 1.000. That is the argument.

    At T = 160 the LSTM's gradient still survives, roughly 1e-08 against the
    RNN's 1e-127, and yet this LSTM ALSO fails at this training budget. It was
    also tried with 220 epochs and a higher forget bias and still failed. So a
    surviving gradient is NECESSARY but not SUFFICIENT: the gates fix signal
    propagation, they do not by themselves make a long dependency easy to find.
    Report both rows. The second one is the more useful lesson.
    """
    if quick:
        lengths, epochs = (10, 90), 60
    rows, curves = [], {}
    for T in lengths:
        Xtr, ytr = make_memory_task(n_train, T, seed=seed)
        Xte, yte = make_memory_task(300, T, seed=seed + 500)
        entry = {"sequence length": T}
        for name, ctor in (("RNN", VanillaRNN), ("LSTM", LSTM)):
            m = ctor(Xtr.shape[2], hidden, 4, seed=seed)
            hist, fg = fit_seq(m, Xtr, ytr, epochs=epochs, lr=0.02, seed=seed)
            entry[f"{name} test acc"] = round(score_seq(m, Xte, yte), 3)
            entry[f"{name} grad at step 1"] = f"{fg[-1]:.2e}"
            curves[(name, T)] = hist
        entry["params RNN"] = VanillaRNN(Xtr.shape[2], hidden, 4).n_params
        entry["params LSTM"] = LSTM(Xtr.shape[2], hidden, 4).n_params
        rows.append(entry)
    return pd.DataFrame(rows), curves


def fig_memory(df, curves):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))

    T = df["sequence length"]
    axes[0].plot(T, df["RNN test acc"], "o-", color=AMBER, lw=2.5, ms=8, label="vanilla RNN")
    axes[0].plot(T, df["LSTM test acc"], "s-", color=SLATE, lw=2.5, ms=8, label="LSTM")
    axes[0].axhline(0.25, color=GREY, ls=":", lw=1.3)
    axes[0].text(T.iloc[0], 0.275, "chance", fontsize=9, color=GREY)
    axes[0].set_xlabel("steps between the symbol and the question")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Remember one symbol across T steps", fontsize=11.5, color=SLATE)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.25)

    longest = int(T.iloc[-1])
    for name, c in (("RNN", AMBER), ("LSTM", SLATE)):
        if (name, longest) in curves:
            axes[1].plot(curves[(name, longest)], color=c, lw=2.2, label=name)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("training loss")
    axes[1].set_title(f"Training loss at T = {longest}\n"
                      "The flat line is not slow learning. It is no learning.",
                      fontsize=11.5, color=SLATE)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.25)

    fig.suptitle("An LSTM's gates are learned control over how fast the past is forgotten",
                 fontsize=12.5, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# The structural comparison that motivates Week 4's third act
# ---------------------------------------------------------------------------

def path_length_table(n_values=(10, 100, 1000)):
    """How many operations separate position 1 from position n?

    This table is the argument for attention, and it is purely structural: no
    training, no data, just counting. It is also the argument for why
    transformers took over, and the second column is why they were POSSIBLE:
    a recurrent network's steps must happen one after another, so its work
    cannot be spread across a GPU. Attention's can.
    """
    rows = []
    for n in n_values:
        rows.append({
            "sequence length": n,
            "RNN path": f"{n} steps",
            "CNN path (k=3)": f"about {int(np.ceil(np.log(n) / np.log(3)))} layers",
            "attention path": "1 step",
            "RNN parallelisable": "no, strictly sequential",
            "attention parallelisable": "yes, all positions at once",
        })
    return pd.DataFrame(rows)


def main(quick=False):
    print("\nEXPERIMENT 1: the gradient through time, measured")
    tbl, rows = gradient_through_time()
    print(tbl.to_string(index=False))
    print("\n  A product of T copies of one matrix. Slightly small vanishes,")
    print("  slightly large explodes, and the usable band is a knife edge.")

    print("\nEXPERIMENT 2: remember the first symbol across T steps")
    df, curves = experiment_memory(quick=quick)
    print(df.to_string(index=False))

    print("\nStructural comparison: how far is position 1 from position n?")
    print(path_length_table().to_string(index=False))
    return tbl, rows, df, curves


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
