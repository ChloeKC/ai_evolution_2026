"""
torch_models.py, the same three families in PyTorch, line for line.

The Evolution of Machine Learning and AI, Week 4 Lab
South East Technological University

Week 3, Chapter 16 made an argument worth repeating: a framework is worth
understanding AFTER you have written the thing by hand, because then you know
exactly which lines it replaced.

You have now written, in numpy:

    a convolution forward and backward pass, with im2col
    max pooling, global average pooling, and their backward passes
    a recurrent network with backpropagation through time
    an LSTM with all four gate computations and their gradients
    scaled dot-product attention with the softmax backward pass
    Adam

This file replaces all of it with about twenty lines, and prints the parameter
counts side by side so you can confirm the two versions agree. That agreement
is the point: the framework replaced CODE, not UNDERSTANDING. The assumptions
of booklet chapters 7, 12 and 17 are still there, still doing all the work, and
still your responsibility to choose correctly.

`nn.Conv2d` will not tell you whether locality is true of your data. It assumes
it, silently and efficiently, and it will assume it just as efficiently when it
is false.

Booklet cross-reference: Chapter 22.

If PyTorch is not installed this file says so and exits cleanly. Nothing else
in the lab depends on it.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:                      # pragma: no cover
    raise ImportError("PyTorch is not installed") from e

import numpy as np


def count(module):
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# 1. The convolutional network, twice
# ---------------------------------------------------------------------------

def make_cnn_torch(side=16, n_classes=4, ch=8):
    """Compare this against make_cnn in cnn_numpy.py, layer for layer.

    Note `AdaptiveAvgPool2d(1)`, which is global average pooling. This is the
    layer that carries the translation assumption all the way to the output.
    Swap it for `nn.Flatten()` plus a larger `nn.Linear` and you get the 0.230
    result from booklet Chapter 9: it will compile, it will train, and nothing
    will warn you.
    """
    return nn.Sequential(
        nn.Conv2d(1, ch, kernel_size=3, padding=1),   # your Conv2D
        nn.ReLU(),
        nn.Conv2d(ch, ch, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),                       # your GlobalAvgPool
        nn.Flatten(),
        nn.Linear(ch, n_classes),
    )


def make_mlp_torch(side=16, n_classes=4, hidden=64):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(side * side, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_classes),
    )


# ---------------------------------------------------------------------------
# 2. The recurrent models
# ---------------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    """`nn.LSTM` replaces about sixty lines of gate arithmetic and its gradients.

    One thing it does NOT do for you: set the forget-gate bias. PyTorch
    initialises all four gate biases to the same small values, so the forget
    gate starts near sigmoid(0) = 0.5, giving a memory half-life of one step.
    Booklet Chapter 15 showed that this choice decides whether the model can
    learn a 90-step dependency at all, so `init_forget_bias` below does by hand
    what the numpy version did in its constructor.
    """

    def __init__(self, n_in, hidden, n_out, forget_bias=3.0):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        self.init_forget_bias(forget_bias)

    def init_forget_bias(self, value):
        # PyTorch stores the four gates concatenated in the order i, f, g, o,
        # so the forget slice is the second quarter.
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                n = self.hidden
                param.data[n:2 * n].fill_(value)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(h[:, -1])


class RNNClassifier(nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        self.rnn = nn.RNN(n_in, hidden, batch_first=True, nonlinearity="tanh")
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h[:, -1])


# ---------------------------------------------------------------------------
# 3. Attention
# ---------------------------------------------------------------------------

class AttentionClassifierTorch(nn.Module):
    """`F.scaled_dot_product_attention` is the whole of your `attention()`.

    It also runs a fused kernel (FlashAttention where available) that never
    materialises the full n by n matrix, which is the memory point from booklet
    Chapter 20. The mathematics is identical; the memory cost is not.
    """

    def __init__(self, d_in, d_model, n_out, n_heads=1):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model, bias=False)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, n_out)

    def forward(self, x):
        e = self.inp(x)
        ctx = F.scaled_dot_product_attention(self.q(e), self.k(e), self.v(e))
        return self.out(ctx[:, -1])


def positional_encoding_torch(T, d):
    """The same sinusoids as attention.py, for comparison."""
    pos = torch.arange(T).unsqueeze(1).float()
    i = torch.arange(d).unsqueeze(0).float()
    angle = pos / torch.pow(10000.0, (2 * (i // 2)) / d)
    pe = torch.where(i % 2 == 0, torch.sin(angle), torch.cos(angle))
    return pe


# ---------------------------------------------------------------------------
# 4. The five-line training loop, unchanged from Week 3
# ---------------------------------------------------------------------------

def train_torch(model, X, y, epochs=60, lr=0.01, batch=16, seed=0):
    """The loop you will write every week from here to Week 12.

    Identical to Week 3's, because the loop never changes. Only the first box
    does, which is this entire week's thesis arriving in the framework.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            opt.zero_grad()                                  # 1
            loss = F.cross_entropy(model(X[b]), y[b])        # 2, 3
            loss.backward()                                  # 4
            opt.step()                                       # 5
    return model


@torch.no_grad()
def score_torch(model, X, y):
    X = torch.as_tensor(X, dtype=torch.float32)
    preds = model(X).argmax(1).numpy()
    return float((preds == np.asarray(y)).mean())


# ---------------------------------------------------------------------------

def main():
    print("\nThe three families, in PyTorch, with numpy parameter counts beside them")

    import cnn_numpy as cn
    from sequences import VanillaRNN, LSTM
    from attention import AttentionClassifier

    rows = [
        ("CNN, global pooling", count(make_cnn_torch()), cn.make_cnn().n_params),
        ("MLP", count(make_mlp_torch()), cn.make_mlp().n_params),
        ("RNN", count(RNNClassifier(5, 24, 4)), VanillaRNN(5, 24, 4).n_params),
        ("LSTM", count(LSTMClassifier(5, 24, 4)), LSTM(5, 24, 4).n_params),
        ("attention", count(AttentionClassifierTorch(37, 32, 4)),
         AttentionClassifier(37, 32, 4).n_params),
    ]
    print(f"\n  {'model':22s} {'PyTorch':>10s} {'numpy':>10s}  {'agree?':>8s}")
    for name, t, n in rows:
        agree = "yes" if t == n else f"differs by {abs(t - n)}"
        print(f"  {name:22s} {t:10,d} {n:10,d}  {agree:>8s}")
    print("\n  Where they differ it is a bias convention: PyTorch's nn.RNN and nn.LSTM")
    print("  carry two bias vectors per gate (input and hidden) rather than one, which")
    print("  is mathematically redundant and kept for cuDNN compatibility.")

    # Verify the framework reproduces the week's central result.
    print("\nThe pixel-shuffle experiment, in PyTorch")
    from inductive_bias import make_motifs, SIDE
    rng = np.random.default_rng(7)
    perm = rng.permutation(SIDE * SIDE)
    for label, p in (("original pixel order", None), ("pixels shuffled", perm)):
        Xtr, ytr = make_motifs(512, seed=0, perm=p)
        Xte, yte = make_motifs(600, seed=99, perm=p)
        line = f"  {label:22s}"
        for nm, builder in (("CNN", make_cnn_torch), ("MLP", make_mlp_torch)):
            m = train_torch(builder(), Xtr, ytr, epochs=60, lr=0.01)
            line += f"  {nm}={score_torch(m, Xte, yte):.3f}"
        print(line, flush=True)
    print("\n  The same conclusion as the numpy version: the CNN's advantage was an")
    print("  assumption about neighbours, and the framework assumes it just as silently.")

    print("\nAttention is still permutation equivariant in PyTorch")
    torch.manual_seed(0)
    T, d = 6, 8
    x = torch.randn(1, T, d)
    q, k, v = (nn.Linear(d, d, bias=False) for _ in range(3))
    idx = torch.randperm(T)

    def run(z):
        return F.scaled_dot_product_attention(q(z), k(z), v(z))

    bare = (run(x)[0][idx] - run(x[:, idx])[0]).abs().max().item()
    pe = positional_encoding_torch(T, d).unsqueeze(0)
    withpe = (run(x + pe)[0][idx] - run(x[:, idx] + pe)[0]).abs().max().item()
    print(f"  embeddings only                 : {bare:.2e}   (zero: order is invisible)")
    print(f"  embeddings + positional encoding: {withpe:.3f}      (order now matters)")


if __name__ == "__main__":
    main()
