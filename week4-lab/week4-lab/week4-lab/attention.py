"""
attention.py, relations as a first-class object.

The Evolution of Machine Learning and AI, Week 4 Lab
South East Technological University

A convolution assumes that what matters is NEARBY. A recurrent network assumes
that what matters arrives in ORDER and can be summarised into a fixed state.
Attention assumes something much weaker and stranger:

    ANY PAIR MIGHT MATTER      every position may be relevant to every other
                               position, and which pairs matter is decided by
                               CONTENT, not by distance or position.

That is a weaker assumption than either of the other two, and weaker
assumptions are usually worse. This one is not, for a specific and
identifiable reason: the pairs that matter in language are not the pairs that
are close together. "The keys that I left on the table in the hall yesterday
are missing" needs to connect "keys" to "are", eleven words apart, and no
amount of locality will find that.

Attention pays for its weak assumption twice. It costs quadratic compute in
the sequence length. And it does not know what order the tokens came in, which
is not a minor gap: it is a mathematical property of the operation, and this
file proves it in EXPERIMENT 2.

    EXPERIMENT 1  Self-attention on a worked example small enough to check by
                  hand: six tokens, printed matrices, one line per stage.

    EXPERIMENT 2  Permutation equivariance. Shuffle the tokens and the outputs
                  come back shuffled the same way, byte for byte. Attention
                  cannot tell "dog bites man" from "man bites dog". This is why
                  positional encoding exists, and it is not a detail bolted on
                  for tidiness.

    EXPERIMENT 3  A content lookup task. Key and value pairs in a sequence,
                  then a query. One attention layer solves it at any distance;
                  an LSTM has to hold every pair in a fixed state and degrades
                  as the sequence grows.

    EXPERIMENT 4  The bill: cost and path length against sequence length.

Booklet cross-references:
    Chapter 17  attention, relations as a first-class object
    Chapter 18  self-attention step by step
    Chapter 19  multi-head attention, positional encoding, the block
    Chapter 20  what the transformer assumes, and what it gives up
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

from optim import AdamMixin

SLATE = "#47546B"
AMBER = "#A96E08"
RED = "#9E3B2C"
GREEN = "#2E7D4F"
BLUE = "#4E7291"
GREY = "#8A93A5"


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# The operation, in five lines
# ---------------------------------------------------------------------------

def attention(Q, K, V, mask=None, return_weights=True):
    """Scaled dot-product attention. This is the whole of it.

        scores  = Q K^T / sqrt(d)      how well does each query match each key
        weights = softmax(scores)      turn the matches into a set of fractions
        out     = weights V            a weighted average of the values

    Read it as a soft dictionary lookup. In a Python dict you supply a key and
    get back exactly one value, and if the key is absent you get an error. Here
    you supply a query, every key reports how well it matches, those match
    scores are turned into fractions that add to 1, and you get back a BLEND of
    the values, weighted by match quality. Nothing is retrieved exactly and
    nothing is missing entirely.

    Why divide by sqrt(d)? Because a dot product of two d-dimensional vectors
    with unit-scale entries grows like sqrt(d). Without the division, scores
    grow with the model's width, softmax saturates on the largest one, and the
    gradient through it goes to nearly zero. It is a scaling fix for a
    numerical problem, and it is the same family of problem as Week 3's
    vanishing gradient.
    """
    d = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    W = softmax(scores, axis=-1)
    out = W @ V
    return (out, W) if return_weights else out


# ---------------------------------------------------------------------------
# EXPERIMENT 1: a worked example you can check with a pencil
# ---------------------------------------------------------------------------

TOKENS = ["the", "keys", "that", "I", "left", "are"]


def worked_example():
    """Six tokens, four dimensions, hand-chosen so the answer is interpretable.

    The embeddings are not learned here. They are written down so that "keys"
    and "are" share a direction, which stands in for grammatical number: both
    are plural. If attention is doing what it claims, the query at "are"
    should put most of its weight on "keys", which is four positions away, and
    very little on "left", which is adjacent.

    A convolution with a width-3 kernel cannot express that preference at all.
    It is not that it would get it wrong; it cannot see position 2 from
    position 6.
    """
    #                      plural  subject  verb   filler
    E = np.array([
        [0.0, 0.1, 0.0, 1.0],      # the
        [1.0, 1.0, 0.0, 0.0],      # keys      <- plural subject
        [0.0, 0.0, 0.0, 1.0],      # that
        [0.2, 0.9, 0.0, 0.1],      # I         <- a competing subject
        [0.0, 0.0, 1.0, 0.2],      # left
        [1.0, 0.0, 1.0, 0.0],      # are       <- plural verb, needs its subject
    ])
    d = E.shape[1]
    # Wq reads "am I a verb looking for a subject"; Wk reads "am I a subject".
    # The factor of 4 is a temperature: it sharpens the softmax. Real models
    # get the same effect by learning larger projection weights.
    Wq = 4.0 * np.array([[1.0, 0, 0, 0], [0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 0]])
    Wk = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    Wv = np.eye(d)

    Q, K, V = E @ Wq, E @ Wk, E @ Wv
    out, W = attention(Q, K, V)

    scores = pd.DataFrame(np.round(Q @ K.T / np.sqrt(d), 3),
                          index=TOKENS, columns=TOKENS)
    weights = pd.DataFrame(np.round(W, 3), index=TOKENS, columns=TOKENS)
    return E, scores, weights, out


def fig_attention_matrix(weights):
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(weights.values, cmap="Blues", vmin=0, vmax=weights.values.max())
    ax.set_xticks(range(len(TOKENS)))
    ax.set_xticklabels(TOKENS, rotation=35, ha="right")
    ax.set_yticks(range(len(TOKENS)))
    ax.set_yticklabels(TOKENS)
    ax.set_xlabel("attended to (keys)")
    ax.set_ylabel("attending from (queries)")
    for i in range(len(TOKENS)):
        for j in range(len(TOKENS)):
            v = weights.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.35 else SLATE)
    ax.add_patch(plt.Rectangle((0.5, 4.5), 1, 1, fill=False, edgecolor=RED, lw=2.6))
    d = len(TOKENS) - 1 - TOKENS.index("keys")
    ax.set_title("Every row adds to 1. Read the bottom row:\n"
                 f"\"are\" spends most of its attention on \"keys\", {d} positions away.",
                 fontsize=11.5, color=SLATE)
    fig.colorbar(im, ax=ax, shrink=0.8, label="attention weight")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# EXPERIMENT 2: attention does not know what order the tokens came in
# ---------------------------------------------------------------------------

def positional_encoding(T, d):
    """The sinusoidal scheme from the original transformer paper.

        PE[t, 2i]   = sin(t / 10000^(2i/d))
        PE[t, 2i+1] = cos(t / 10000^(2i/d))

    Why sinusoids rather than just writing the number t into a slot? Because a
    plain counter grows without bound, so position 5000 would swamp every
    semantic dimension. Sinusoids stay in the range minus 1 to 1 at every
    position, and different dimensions oscillate at wildly different rates, so
    the pattern across dimensions is unique per position, like the hands of a
    set of clocks running at different speeds. It also gives the model an easy
    route to RELATIVE position: a fixed offset is a fixed rotation.

    Modern models mostly use learned or rotary encodings instead. The reason
    they all exist is the same, and it is the number printed by
    permutation_test() below.
    """
    pos = np.arange(T)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d)
    PE = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
    return PE


def permutation_test(T=6, d=8, seed=0):
    """Shuffle the tokens. Do the outputs shuffle identically, or change?

    If shuffling the input merely shuffles the output, the operation has no
    notion of order: it is computing a function of the SET of tokens. That is
    called permutation equivariance, and for attention it is exact, not
    approximate. The number below is zero to machine precision, and that zero
    is the reason every transformer ever shipped adds position information
    before the first attention layer.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (T, d))
    Wq, Wk, Wv = (rng.normal(0, 0.5, (d, d)) for _ in range(3))
    perm = rng.permutation(T)

    def run(Z):
        return attention(Z @ Wq, Z @ Wk, Z @ Wv, return_weights=False)

    out_then_perm = run(X)[perm]
    perm_then_out = run(X[perm])
    bare = np.abs(out_then_perm - perm_then_out).max()

    PE = positional_encoding(T, d)
    out_then_perm_pe = run(X + PE)[perm]
    perm_then_out_pe = run(X[perm] + PE)
    withpe = np.abs(out_then_perm_pe - perm_then_out_pe).max()

    return pd.DataFrame([
        {"input": "embeddings only",
         "max |shuffle(attend(x)) - attend(shuffle(x))|": f"{bare:.2e}",
         "verdict": "order is invisible: a bag of tokens"},
        {"input": "embeddings + positional encoding",
         "max |shuffle(attend(x)) - attend(shuffle(x))|": f"{withpe:.3f}",
         "verdict": "order now changes the answer"},
    ])


def fig_positional_encoding(T=48, d=32):
    PE = positional_encoding(T, d)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    im = axes[0].imshow(PE.T, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_xlabel("position in the sequence")
    axes[0].set_ylabel("embedding dimension")
    axes[0].set_title("Each column is a position's fingerprint", fontsize=11, color=SLATE)
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    for dim, c in ((0, SLATE), (4, AMBER), (12, GREEN), (24, RED)):
        axes[1].plot(PE[:, dim], lw=2.1, color=c, label=f"dimension {dim}")
    axes[1].set_xlabel("position")
    axes[1].set_ylabel("value")
    axes[1].set_title("Clocks running at different speeds\nso no two positions look alike",
                      fontsize=11, color=SLATE)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)
    fig.suptitle("Positional encoding exists because attention is permutation equivariant",
                 fontsize=12.5, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-head attention: several relations at once
# ---------------------------------------------------------------------------

def multi_head(X, Wq, Wk, Wv, n_heads):
    """Split the width into n_heads slices and attend independently in each.

    A single attention layer produces ONE set of weights per position, so it
    must express every relationship it cares about in a single weighted
    average. Language does not work like that: a verb needs its subject, its
    object, its tense marker and its clause boundary, and blending those into
    one average loses all of them.

    Multi-head attention is the obvious fix and it costs nothing extra: use
    the same total width, cut into h independent slices, and let each slice
    learn its own notion of relevance. Trained transformers reliably develop
    heads that specialise, and some are interpretable enough to name.
    """
    T, d = X.shape
    dh = d // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    outs, weights = [], []
    for h in range(n_heads):
        s = slice(h * dh, (h + 1) * dh)
        o, w = attention(Q[:, s], K[:, s], V[:, s])
        outs.append(o)
        weights.append(w)
    return np.concatenate(outs, axis=1), np.stack(weights)


def fig_multi_head(T=10, d=16, n_heads=4, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (T, d)) + positional_encoding(T, d)
    Wq, Wk, Wv = (rng.normal(0, 0.6, (d, d)) for _ in range(3))
    _, W = multi_head(X, Wq, Wk, Wv, n_heads)
    fig, axes = plt.subplots(1, n_heads, figsize=(13.2, 3.6))
    for h, ax in enumerate(axes):
        ax.imshow(W[h], cmap="Blues", vmin=0, vmax=W[h].max())
        ax.set_title(f"head {h + 1}", fontsize=10.5, color=SLATE)
        ax.set_xlabel("attended to")
        if h == 0:
            ax.set_ylabel("attending from")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Four heads, one layer, untrained random weights: already four different "
                 "notions of what is relevant",
                 fontsize=12, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# EXPERIMENT 3: a trainable single-head attention layer
# ---------------------------------------------------------------------------

class AttentionClassifier(AdamMixin):
    """One attention layer, then a linear readout at the final position.

    Written out with its backward pass so that nothing is hidden. The softmax
    backward is the only unfamiliar piece:

        if p = softmax(s), then  ds = p * (dp - sum(dp * p))

    which is the standard Jacobian of a softmax applied without ever building
    the Jacobian, exactly as Week 3's cross-entropy gradient collapsed to
    p minus y.
    """

    def __init__(self, d_in, d_model, n_out, seed=0, use_pe=True):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_in)
        self.Wi = rng.normal(0, s, (d_in, d_model))
        self.Wq = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
        self.Wk = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
        self.Wv = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
        self.Wo = rng.normal(0, 1 / np.sqrt(d_model), (d_model, n_out))
        self.bo = np.zeros(n_out)
        self.d_model, self.use_pe = d_model, use_pe
        self.params = ["Wi", "Wq", "Wk", "Wv", "Wo", "bo"]

    @property
    def n_params(self):
        return sum(getattr(self, p).size for p in self.params)

    def forward(self, X):
        N, T, _ = X.shape
        E = X @ self.Wi
        if self.use_pe:
            E = E + positional_encoding(T, self.d_model)[None, :, :]
        Q, K, V = E @ self.Wq, E @ self.Wk, E @ self.Wv
        scores = Q @ np.swapaxes(K, 1, 2) / np.sqrt(self.d_model)
        W = softmax(scores, axis=-1)
        ctx = W @ V
        self.cache = (X, E, Q, K, V, W, ctx)
        self.last_weights = W
        return ctx[:, -1] @ self.Wo + self.bo

    def backward(self, d_logits):
        X, E, Q, K, V, W, ctx = self.cache
        N, T, dm = E.shape

        self.dWo = ctx[:, -1].T @ d_logits
        self.dbo = d_logits.sum(0)

        d_ctx = np.zeros_like(ctx)
        d_ctx[:, -1] = d_logits @ self.Wo.T

        dW = d_ctx @ np.swapaxes(V, 1, 2)
        dV = np.swapaxes(W, 1, 2) @ d_ctx

        # softmax backward, row by row, without materialising a Jacobian
        d_scores = W * (dW - (dW * W).sum(axis=-1, keepdims=True))
        d_scores /= np.sqrt(dm)

        dQ = d_scores @ K
        dK = np.swapaxes(d_scores, 1, 2) @ Q

        self.dWq = np.einsum("ntd,nte->de", E, dQ)
        self.dWk = np.einsum("ntd,nte->de", E, dK)
        self.dWv = np.einsum("ntd,nte->de", E, dV)

        dE = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        self.dWi = np.einsum("ntd,nte->de", X, dE)


def gradient_check(seed=0, eps=1e-6):
    """A wrong derivative fails silently. Week 3, Chapter 14, still true."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (3, 7, 9))
    y = rng.integers(0, 4, 3)
    m = AttentionClassifier(9, 8, 4, seed=seed)

    def loss_of():
        lg = m.forward(X)
        p = softmax(lg)
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()

    lg = m.forward(X)
    p = softmax(lg)
    d = p.copy()
    d[np.arange(len(y)), y] -= 1
    m.backward(d / len(y))

    worst = 0.0
    for name in ("Wq", "Wk", "Wv", "Wi"):
        Wm = getattr(m, name)
        an = getattr(m, "d" + name)
        for _ in range(25):
            idx = tuple(rng.integers(0, s) for s in Wm.shape)
            old = Wm[idx]
            Wm[idx] = old + eps
            lp = loss_of()
            Wm[idx] = old - eps
            lm = loss_of()
            Wm[idx] = old
            num = (lp - lm) / (2 * eps)
            rel = abs(an[idx] - num) / (abs(an[idx]) + abs(num) + 1e-12)
            worst = max(worst, rel)
    return worst


def make_lookup_task(n, T, n_keys=32, n_vals=4, seed=0):
    """A sequence of key and value pairs, then a query. Answer: the matching value.

    Each of the first T-1 positions holds one (key, value) pair. The final
    position holds a query key and no value. The label is the value that was
    paired with that key.

    Nothing about this task depends on distance. The answer might sit at
    position 2 or position 40, and a correct model should not care. That is
    precisely the property attention has and recurrence does not: the LSTM has
    to carry every pair it has seen in a fixed-size state, so as T grows the
    state has to hold more, and a fixed vector cannot.
    """
    rng = np.random.default_rng(seed)
    n_pairs = T - 1
    if n_pairs > n_keys:
        raise ValueError("need at least T-1 distinct keys for the query to be unambiguous")
    d_in = n_keys + n_vals + 1
    X = np.zeros((n, T, d_in))
    y = np.zeros(n, dtype=int)
    for i in range(n):
        # Keys are drawn WITHOUT replacement, so exactly one pair matches the
        # query. If keys could repeat the label would be ambiguous and no model,
        # however good, could exceed chance on the repeated ones.
        keys = rng.permutation(n_keys)[:n_pairs]
        vals = rng.integers(0, n_vals, n_pairs)
        for t in range(n_pairs):
            X[i, t, keys[t]] = 1.0
            X[i, t, n_keys + vals[t]] = 1.0
        pick = rng.integers(0, n_pairs)
        X[i, -1, keys[pick]] = 1.0
        X[i, -1, -1] = 1.0                     # the "this is the query" flag
        y[i] = vals[pick]
    return X, y


def experiment_lookup(lengths=(4, 9, 17, 25), n_train=600, epochs=50,
                      seed=0, quick=False):
    """Attention against an LSTM on content lookup, as the distance grows."""
    from sequences import LSTM, fit_seq, score_seq, ce_loss

    if quick:
        lengths, epochs = (4, 25), 30
    rows = []
    for T in lengths:
        Xtr, ytr = make_lookup_task(n_train, T, seed=seed)
        Xte, yte = make_lookup_task(400, T, seed=seed + 700)
        entry = {"sequence length": T, "pairs to remember": T - 1}

        att = AttentionClassifier(Xtr.shape[2], 32, 4, seed=seed, use_pe=False)
        _fit_attention(att, Xtr, ytr, epochs=epochs, seed=seed)
        entry["attention test acc"] = round(_score(att, Xte, yte), 3)
        entry["attention params"] = att.n_params

        # The LSTM gets a MORE generous budget than the attention model here,
        # deliberately. It was also tried at lr 0.03 for 150 epochs and did no
        # better: at T=25 it scored 0.270 instead of 0.278. This is not
        # undertraining, it is a structural mismatch. A fixed-size state has to
        # hold all T-1 pairs at once, and the number of pairs grows while the
        # state does not.
        lstm = LSTM(Xtr.shape[2], 32, 4, seed=seed)
        fit_seq(lstm, Xtr, ytr, epochs=max(epochs, 100), lr=0.02, seed=seed)
        entry["LSTM test acc"] = round(score_seq(lstm, Xte, yte), 3)
        entry["LSTM params"] = lstm.n_params
        rows.append(entry)
    return pd.DataFrame(rows)


def _fit_attention(m, X, y, epochs=60, lr=0.01, batch=32, seed=0):
    rng = np.random.default_rng(seed)
    hist = []
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        tot = 0.0
        for s in range(0, len(X), batch):
            b = idx[s:s + batch]
            lg = m.forward(X[b])
            p = softmax(lg)
            loss = -np.log(p[np.arange(len(b)), y[b]] + 1e-12).mean()
            d = p.copy()
            d[np.arange(len(b)), y[b]] -= 1
            m.backward(d / len(b))
            m.step(lr)
            tot += loss * len(b)
        hist.append(tot / len(X))
    return hist


def _score(m, X, y, batch=128):
    preds = []
    for s in range(0, len(X), batch):
        preds.append(m.forward(X[s:s + batch]).argmax(1))
    return float((np.concatenate(preds) == y).mean())


def fig_learned_lookup(T=13, seed=0):
    """What the trained attention layer actually looks at. The proof of concept."""
    Xtr, ytr = make_lookup_task(600, T, seed=seed)
    m = AttentionClassifier(Xtr.shape[2], 32, 4, seed=seed, use_pe=False)
    _fit_attention(m, Xtr, ytr, epochs=50, seed=seed)

    Xte, yte = make_lookup_task(1, T, seed=4242)
    m.forward(Xte)
    W = m.last_weights[0]
    n_keys = 32
    query_key = int(np.argmax(Xte[0, -1, :n_keys]))
    match = [t for t in range(T - 1) if np.argmax(Xte[0, t, :n_keys]) == query_key]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    im = axes[0].imshow(W, cmap="Blues", vmin=0)
    axes[0].set_xlabel("attended to (position)")
    axes[0].set_ylabel("attending from (position)")
    axes[0].set_title("The learned attention pattern", fontsize=11, color=SLATE)
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    axes[1].bar(range(T), W[-1], color=[RED if t in match else SLATE for t in range(T)])
    axes[1].set_xlabel("position")
    axes[1].set_ylabel("attention weight from the query")
    axes[1].set_title(f"The query's row.\nThe matching pair is at position {match[0]}, "
                      "in red.", fontsize=11, color=SLATE)
    axes[1].grid(alpha=0.25, axis="y")
    fig.suptitle("Nobody wrote a lookup rule. The weights that implement one were learned.",
                 fontsize=12.5, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# EXPERIMENT 4: the bill
# ---------------------------------------------------------------------------

def cost_table(lengths=(64, 512, 4096, 32768), d=512, k=3):
    """What each family costs per layer, and how far apart two positions are.

    Read the last two columns together. Attention buys a path length of 1 at
    any distance, which is what makes long-range relations learnable at all.
    It pays for it in operations that grow with the SQUARE of the length, which
    is why context windows were small for years and why a great deal of current
    research is about making this table less ugly.
    """
    rows = []
    for n in lengths:
        rnn = n * d * d
        cnn = n * k * d * d
        att = n * n * d + n * d * d
        rows.append({
            "sequence length": n,
            "RNN ops": f"{rnn:.2e}",
            "CNN ops": f"{cnn:.2e}",
            "attention ops": f"{att:.2e}",
            "attention / RNN": f"{att / rnn:.1f}x",
            "longest path (attention)": 1,
            "longest path (RNN)": n,
        })
    return pd.DataFrame(rows)


def fig_cost(lengths=(64, 256, 1024, 4096, 16384), d=512, k=3):
    n = np.array(lengths, float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].loglog(n, n * d * d, "o-", color=AMBER, lw=2.3, label="RNN, O(n d^2)")
    axes[0].loglog(n, n * k * d * d, "^-", color=GREEN, lw=2.3, label="CNN, O(n k d^2)")
    axes[0].loglog(n, n * n * d + n * d * d, "s-", color=SLATE, lw=2.3,
                   label="attention, O(n^2 d + n d^2)")
    axes[0].set_xlabel("sequence length")
    axes[0].set_ylabel("operations per layer")
    axes[0].set_title("What it costs", fontsize=11.5, color=SLATE)
    axes[0].legend(fontsize=9.5)
    axes[0].grid(alpha=0.25, which="both")

    axes[1].loglog(n, n, "o-", color=AMBER, lw=2.3, label="RNN: n steps")
    axes[1].loglog(n, np.ceil(np.log(n) / np.log(k)), "^-", color=GREEN, lw=2.3,
                   label="CNN: log_k(n) layers")
    axes[1].loglog(n, np.ones_like(n), "s-", color=SLATE, lw=2.3, label="attention: 1 step")
    axes[1].set_xlabel("sequence length")
    axes[1].set_ylabel("operations between position 1 and position n")
    axes[1].set_title("What it buys", fontsize=11.5, color=SLATE)
    axes[1].legend(fontsize=9.5)
    axes[1].grid(alpha=0.25, which="both")
    fig.suptitle("The transformer's bargain, in two plots",
                 fontsize=12.5, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


def main(quick=False):
    print("\nEXPERIMENT 1: self-attention on six tokens")
    E, scores, weights, out = worked_example()
    print("\n  Raw scores, Q K^T / sqrt(d):")
    print(scores.to_string())
    print("\n  After softmax, every row adds to 1:")
    print(weights.to_string())
    row = weights.loc["are"]
    d = len(TOKENS) - 1 - TOKENS.index("keys")
    print(f"\n  The query at \"are\" gives {row['keys']:.3f} to \"keys\" ({d} positions away)")
    print(f"  and {row['left']:.3f} to \"left\" (adjacent). Distance did not decide it.")

    print("\nEXPERIMENT 2: does attention know what order the tokens came in?")
    print(permutation_test().to_string(index=False))

    print("\nGradient check on the hand-written attention backward pass")
    print(f"  max relative error: {gradient_check():.3e}")

    print("\nEXPERIMENT 3: content lookup, attention against an LSTM")
    look = experiment_lookup(quick=quick)
    print(look.to_string(index=False))

    print("\nEXPERIMENT 4: the bill")
    print(cost_table().to_string(index=False))
    return scores, weights, look


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
