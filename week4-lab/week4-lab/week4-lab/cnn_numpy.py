"""
cnn_numpy.py, a convolutional network with every derivative written out.

The Evolution of Machine Learning and AI, Week 4 Lab
South East Technological University

In Week 3 you wrote backpropagation for a dense network in about eighty lines.
This is the same exercise for a convolutional one, and the point of doing it
is that a convolution layer's backward pass makes its assumption visible.

Look at `Conv2D.backward`. The gradient for the kernel is a SUM over every
position the kernel visited. Every patch in the image contributes to the same
nine numbers. That summation IS weight sharing, and weight sharing is the
stationarity assumption from `convolution.py`, expressed as calculus. It is
also why a CNN extracts more learning signal per training image than a dense
network does: one image gives you hundreds of gradient contributions to the
same small set of weights, instead of one contribution to each of many.

Booklet cross-references:
    Chapter 5   weight sharing and parameter counts
    Chapter 7   what a convolution layer assumes
    Chapter 10  CNN architectures, and what each one fixed

Nothing here imports a deep learning library.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# im2col: the trick that makes convolution a matrix multiply
# ---------------------------------------------------------------------------
# A convolution is a lot of small dot products. A GPU is very fast at one big
# matrix multiply and comparatively slow at many small ones. im2col rearranges
# every patch the kernel will visit into a column of one large matrix, so the
# whole layer becomes a single matmul. This is not a mathematical insight, it
# is a hardware one, and it is the same argument as Week 3's mini-batch: the
# operation that wins is the operation that suits the silicon.

def im2col(X, kh, kw, stride=1, pad=0):
    """X: (N, C, H, W) -> (N * out_h * out_w, C * kh * kw)."""
    N, C, H, W = X.shape
    if pad:
        X = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    s = X.strides
    shape = (N, C, out_h, out_w, kh, kw)
    strides = (s[0], s[1], s[2] * stride, s[3] * stride, s[2], s[3])
    patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    cols = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * out_h * out_w, -1)
    return np.ascontiguousarray(cols), out_h, out_w


def col2im(cols, X_shape, kh, kw, stride=1, pad=0, out_h=None, out_w=None):
    """The reverse of im2col, ACCUMULATING overlaps.

    The accumulation is not a detail. A pixel that took part in nine different
    patches receives nine gradient contributions, and they must be added. That
    addition is the mirror image of weight sharing in the forward pass.
    """
    N, C, H, W = X_shape
    Hp, Wp = H + 2 * pad, W + 2 * pad
    out = np.zeros((N, C, Hp, Wp))
    cols = cols.reshape(N, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    for i in range(kh):
        for j in range(kw):
            out[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride] += \
                cols[:, :, i, j, :, :]
    return out[:, :, pad:pad + H, pad:pad + W] if pad else out


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class Conv2D:
    """out_channels filters, each (in_channels, k, k). The parameters do not
    depend on the image size at all, which is the whole economic argument."""

    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=1, rng=None):
        rng = rng or np.random.default_rng(0)
        # He initialisation: variance 2 / fan_in. Week 3's exercise 3 asked
        # what happens at scale 50 and at scale 0; this is the principled
        # middle, and it matters far more in a deep net than a shallow one.
        fan_in = in_ch * k * k
        self.W = rng.normal(0, np.sqrt(2.0 / fan_in), (out_ch, in_ch, k, k))
        self.b = np.zeros(out_ch)
        self.k, self.stride, self.pad = k, stride, pad

    @property
    def n_params(self):
        return self.W.size + self.b.size

    def forward(self, X):
        self.X_shape = X.shape
        cols, oh, ow = im2col(X, self.k, self.k, self.stride, self.pad)
        self.cols, self.oh, self.ow = cols, oh, ow
        Wf = self.W.reshape(self.W.shape[0], -1)          # (out_ch, C*k*k)
        out = cols @ Wf.T + self.b
        N = X.shape[0]
        return out.reshape(N, oh, ow, -1).transpose(0, 3, 1, 2)

    def backward(self, d_out):
        N = d_out.shape[0]
        d_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, d_out.shape[1])

        # THE LINE THAT IS WEIGHT SHARING. d_flat has one row per (image,
        # position). The matmul sums over ALL of them, so every position the
        # kernel visited contributes to the same nine numbers per channel.
        self.dW = (d_flat.T @ self.cols).reshape(self.W.shape)
        self.db = d_flat.sum(axis=0)

        Wf = self.W.reshape(self.W.shape[0], -1)
        d_cols = d_flat @ Wf
        return col2im(d_cols, self.X_shape, self.k, self.k,
                      self.stride, self.pad, self.oh, self.ow)

    def step(self, lr, m=0.9):
        if not hasattr(self, "vW"):
            self.vW = np.zeros_like(self.W)
            self.vb = np.zeros_like(self.b)
        self.vW = m * self.vW - lr * self.dW
        self.vb = m * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


class MaxPool:
    """Take the largest value in each window and throw the rest away.

    Two things happen at once and it is worth separating them. The output
    shrinks, which saves compute. And the exact position of the winning value
    is discarded, which buys a little INVARIANCE: a feature that moves by one
    pixel inside a window produces the same output. Convolution gives
    equivariance (move the input, the output moves); pooling converts some of
    that into invariance (move the input a little, the output does not change).
    """

    def __init__(self, k=2):
        self.k = k
        self.n_params = 0

    def forward(self, X):
        N, C, H, W = X.shape
        k = self.k
        H2, W2 = H // k, W // k
        Xc = X[:, :, :H2 * k, :W2 * k].reshape(N, C, H2, k, W2, k)
        out = Xc.max(axis=(3, 5))
        self.mask = (Xc == out[:, :, :, None, :, None])
        self.X_shape, self.shapes = X.shape, (N, C, H2, W2)
        return out

    def backward(self, d_out):
        N, C, H2, W2 = self.shapes
        k = self.k
        d = self.mask * d_out[:, :, :, None, :, None]
        d = d.reshape(N, C, H2 * k, W2 * k)
        full = np.zeros(self.X_shape)
        full[:, :, :H2 * k, :W2 * k] = d
        return full

    def step(self, *a, **kw):
        pass


class GlobalAvgPool:
    """Average each channel over the whole feature map. One number per channel.

    This layer is where equivariance becomes INVARIANCE, and the distinction
    is the most commonly missed point in the week.

    A convolution is equivariant: move the input, and the feature map moves
    with it. That is useful but it is not the same as not caring. If you
    finish a CNN with Flatten and a dense layer, that dense layer reads
    positions, so a feature that moved is a feature in a different input slot,
    and the network's answer changes. The convolutional trunk kept the
    assumption; the head threw it away.

    Averaging over position asks "is this feature present anywhere?" and
    discards where. Now moving the input does not change the output at all.
    The assumption is carried all the way to the prediction, which is what you
    actually wanted. This is also why the parameter count stops depending on
    the input size, and therefore why the same trained network can be applied
    to a larger image.
    """

    def __init__(self):
        self.n_params = 0

    def forward(self, X):
        self.shape = X.shape
        return X.mean(axis=(2, 3))

    def backward(self, d):
        N, C, H, W = self.shape
        return np.broadcast_to(d[:, :, None, None] / (H * W), self.shape).copy()

    def step(self, *a, **kw):
        pass


class ReLU:
    def __init__(self):
        self.n_params = 0

    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, d):
        return d * self.mask

    def step(self, *a, **kw):
        pass


class Flatten:
    def __init__(self):
        self.n_params = 0

    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, d):
        return d.reshape(self.shape)

    def step(self, *a, **kw):
        pass


class Dense:
    def __init__(self, n_in, n_out, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)

    @property
    def n_params(self):
        return self.W.size + self.b.size

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, d):
        self.dW = self.X.T @ d
        self.db = d.sum(axis=0)
        return d @ self.W.T

    def step(self, lr, m=0.9):
        if not hasattr(self, "vW"):
            self.vW = np.zeros_like(self.W)
            self.vb = np.zeros_like(self.b)
        self.vW = m * self.vW - lr * self.dW
        self.vb = m * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


# ---------------------------------------------------------------------------
# Loss and model
# ---------------------------------------------------------------------------

def softmax_ce(logits, y):
    """Softmax cross-entropy, and its gradient, which is beautifully simple."""
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p.copy()
    d[np.arange(n), y] -= 1.0
    return loss, d / n


class Net:
    """A list of layers, run forwards then backwards. That is all a network is."""

    def __init__(self, layers):
        self.layers = layers

    @property
    def n_params(self):
        return sum(l.n_params for l in self.layers)

    def forward(self, X):
        for l in self.layers:
            X = l.forward(X)
        return X

    def backward(self, d):
        for l in reversed(self.layers):
            d = l.backward(d)

    def step(self, lr):
        for l in self.layers:
            l.step(lr)

    def fit(self, X, y, epochs=30, lr=0.05, batch=32, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        n = len(X)
        history = []
        for ep in range(epochs):
            idx = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                logits = self.forward(X[b])
                loss, d = softmax_ce(logits, y[b])
                self.backward(d)
                self.step(lr)
                tot += loss * len(b)
            history.append(tot / n)
            if verbose and (ep + 1) % 10 == 0:
                print(f"    epoch {ep+1:3d}  loss {history[-1]:.4f}")
        return history

    def score(self, X, y, batch=256):
        preds = []
        for s in range(0, len(X), batch):
            preds.append(self.forward(X[s:s + batch]).argmax(axis=1))
        return float((np.concatenate(preds) == y).mean())


# ---------------------------------------------------------------------------
# The two architectures being compared all week
# ---------------------------------------------------------------------------

def make_cnn(side=16, n_classes=4, ch=8, seed=0):
    """Locality and weight sharing assumed, and carried through to the output.

    Two convolutions, then GLOBAL average pooling. The question this network
    can ask is "is this feature present anywhere in the image?", and it cannot
    ask "is this feature at position 47", because it has thrown that away on
    purpose.
    """
    rng = np.random.default_rng(seed)
    return Net([
        Conv2D(1, ch, k=3, pad=1, rng=rng),
        ReLU(),
        Conv2D(ch, ch, k=3, pad=1, rng=rng),
        ReLU(),
        GlobalAvgPool(),
        Dense(ch, n_classes, rng=rng),
    ])


def make_cnn_flatten(side=16, n_classes=4, ch=8, seed=0):
    """The same convolutional trunk, with a position-reading head bolted on.

    Included because the comparison against make_cnn is instructive and
    slightly uncomfortable: two networks with identical convolutional layers
    behave completely differently on shifted inputs, because one of them
    discards the assumption in its final layer.
    """
    rng = np.random.default_rng(seed)
    return Net([
        Conv2D(1, ch, k=3, pad=1, rng=rng),
        ReLU(),
        Conv2D(ch, ch, k=3, pad=1, rng=rng),
        ReLU(),
        MaxPool(2),
        Flatten(),
        Dense(ch * (side // 2) * (side // 2), n_classes, rng=rng),
    ])


def make_mlp(side=16, n_classes=4, hidden=64, seed=0):
    """No assumption about space at all. Every pixel is just a number in a list.

    Worth being clear that this is not a straw man. This network is strictly
    more general than the CNN: the set of functions it can represent includes
    every function the CNN can represent. It is more powerful and it does
    worse, and understanding why is the whole of Week 4.
    """
    rng = np.random.default_rng(seed)
    return Net([
        Flatten(),
        Dense(side * side, hidden, rng=rng),
        ReLU(),
        Dense(hidden, n_classes, rng=rng),
    ])


# ---------------------------------------------------------------------------
# Gradient check, exactly as in Week 3. A wrong derivative fails silently.
# ---------------------------------------------------------------------------

def gradient_check(seed=0, eps=3e-5):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (4, 1, 16, 16))
    y = rng.integers(0, 4, 4)
    net = make_cnn(side=16, n_classes=4, seed=seed)

    logits = net.forward(X)
    loss, d = softmax_ce(logits, y)
    net.backward(d)

    conv = net.layers[0]
    analytic = conv.dW.copy()
    numeric = np.zeros_like(analytic)
    it = np.ndindex(*analytic.shape)
    for idx in it:
        old = conv.W[idx]
        conv.W[idx] = old + eps
        lp, _ = softmax_ce(net.forward(X), y)
        conv.W[idx] = old - eps
        lm, _ = softmax_ce(net.forward(X), y)
        conv.W[idx] = old
        numeric[idx] = (lp - lm) / (2 * eps)

    denom = np.abs(analytic) + np.abs(numeric) + 1e-12
    rel = np.abs(analytic - numeric) / denom
    return float(rel.max()), float(loss)


if __name__ == "__main__":
    rel, loss = gradient_check()
    print(f"Conv2D gradient check: max relative error {rel:.3e}  (initial loss {loss:.4f})")
    print("Below about 1e-5 means the hand-written convolution backward pass is correct.")
    print("The floor here is the CENTRAL DIFFERENCE, not the derivative: at eps=1e-4 the")
    print("error rises to 2e-2 through rounding, and at eps=1e-5 it rises again. There is")
    print("a sweet spot, and finding it is part of knowing what the check can tell you.")
    cnn, mlp = make_cnn(), make_mlp()
    print(f"\nCNN parameters: {cnn.n_params:,}")
    print(f"MLP parameters: {mlp.n_params:,}")
