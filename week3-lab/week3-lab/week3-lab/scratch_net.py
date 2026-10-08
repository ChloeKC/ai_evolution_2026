"""
scratch_net.py, backpropagation, written out, solving the problem that
froze the field for seventeen years.

The Evolution of Machine Learning and AI · Week 3 Lab
South East Technological University

Week 1, Chapter 6: Minsky and Papert proved in 1969 that a single-layer
perceptron cannot compute XOR. The proof was correct. The escape route was a
hidden layer, and the field could not train one until backpropagation was
popularised in 1986.

This file is that seventeen-year gap, in about eighty lines. First we watch the
perceptron fail, exactly as promised. Then we add one hidden layer, write the
chain rule out by hand, and watch it succeed.

Booklet cross-references:
    Chapter 7   the perceptron and its wall
    Chapter 11  backpropagation as credit assignment
    Chapter 12  why depth needed both an algorithm and a machine

Nothing here imports a deep learning library. That is the point: when you meet
`loss.backward()` in torch_intro.py, you will know exactly which four lines of
this file it replaced.
"""

from __future__ import annotations

import numpy as np

# XOR: the four points that ended the first neural network era.
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = np.array([[0.], [1.], [1.], [0.]])


# ---------------------------------------------------------------------------
# 1969: the single-layer perceptron, and the wall
# ---------------------------------------------------------------------------

def perceptron(X, y, lr=0.1, epochs=200, seed=0):
    """Rosenblatt's rule: w <- w + lr * (target - prediction) * x

    Guaranteed to converge IF the data is linearly separable. XOR is not, so
    this loop will run forever without ever getting all four points right.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.5, X.shape[1])
    b = 0.0
    history = []
    for _ in range(epochs):
        for xi, target in zip(X, y.ravel()):
            pred = 1.0 if (xi @ w + b) > 0 else 0.0
            err = target - pred
            w += lr * err * xi
            b += lr * err
        preds = ((X @ w + b) > 0).astype(float)
        history.append(float((preds == y.ravel()).mean()))
    return w, b, np.array(history)


# ---------------------------------------------------------------------------
# 1986: one hidden layer, and the chain rule
# ---------------------------------------------------------------------------

def sigmoid(z):     return 1.0 / (1.0 + np.exp(-z))
def sigmoid_d(a):   return a * (1.0 - a)          # in terms of the OUTPUT a


class TinyNet:
    """A 2 -> h -> 1 network with every derivative written out by hand.

    Layout, and the names used in the booklet:

        z1 = X W1 + b1        a1 = sigma(z1)        hidden pre-activation, activation
        z2 = a1 W2 + b2       a2 = sigma(z2)        output pre-activation, prediction
        L  = mean (a2 - y)^2                        the loss
    """

    def __init__(self, hidden=2, seed=0):
        rng = np.random.default_rng(seed)
        # Scale matters: too small and every unit starts identical and stays
        # identical; too large and sigmoid saturates and the gradient vanishes.
        self.W1 = rng.normal(0, 1.0, (2, hidden))
        self.b1 = np.zeros((1, hidden))
        self.W2 = rng.normal(0, 1.0, (hidden, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, Y, lr):
        """The whole of backpropagation. Four gradients, three chain rules.

        dL/da2 = 2(a2 - y)/n            how wrong the output is
        dL/dz2 = dL/da2 * sigma'(a2)    push it back through the activation
        dL/dW2 = a1^T @ dL/dz2          blame the weights that fed it
        dL/da1 = dL/dz2 @ W2^T          send the blame BACK a layer  <- the trick
        """
        n = len(X)
        d_a2 = 2.0 * (self.a2 - Y) / n
        d_z2 = d_a2 * sigmoid_d(self.a2)

        d_W2 = self.a1.T @ d_z2
        d_b2 = d_z2.sum(axis=0, keepdims=True)

        d_a1 = d_z2 @ self.W2.T                  # credit assignment, backwards
        d_z1 = d_a1 * sigmoid_d(self.a1)

        d_W1 = X.T @ d_z1
        d_b1 = d_z1.sum(axis=0, keepdims=True)

        for param, grad in ((self.W1, d_W1), (self.b1, d_b1),
                            (self.W2, d_W2), (self.b2, d_b2)):
            param -= lr * grad               # the one line from Chapter 8

    def fit(self, X, Y, lr=0.5, epochs=8000, log_every=2000):
        history = []
        for e in range(epochs + 1):
            out = self.forward(X)
            loss = float(np.mean((out - Y) ** 2))
            history.append(loss)
            if log_every and e % log_every == 0:
                print(f"    epoch {e:>5}   loss {loss:.6f}")
            self.backward(X, Y, lr)
        return np.array(history)


def gradient_check(net, X, Y, eps=1e-6):
    """Numerically verify one hand-derived gradient. Always do this once.

    Perturb a single weight by eps, see how the loss actually changes, and
    compare against what backward() claimed. If they disagree, your derivation
    is wrong and no amount of training will save you.
    """
    net.forward(X)
    net_copy_W2 = net.W2.copy()
    n = len(X)
    d_a2 = 2.0 * (net.a2 - Y) / n
    d_z2 = d_a2 * sigmoid_d(net.a2)
    analytic = (net.a1.T @ d_z2)[0, 0]

    def loss_at(w):
        net.W2 = net_copy_W2.copy()
        net.W2[0, 0] = w
        return float(np.mean((net.forward(X) - Y) ** 2))

    w0 = net_copy_W2[0, 0]
    numeric = (loss_at(w0 + eps) - loss_at(w0 - eps)) / (2 * eps)
    net.W2 = net_copy_W2
    return analytic, numeric


if __name__ == "__main__":
    print("=" * 70)
    print("1969 · the single-layer perceptron meets XOR")
    print("=" * 70)
    w, b, hist = perceptron(X, Y)
    print(f"  best accuracy over 200 epochs: {hist.max():.2f}")
    print(f"  final accuracy:                {hist[-1]:.2f}")
    print("  It never exceeds 3 of 4. There is no straight line that works,")
    print("  so there is no weight vector for the rule to find.\n")

    print("=" * 70)
    print("1986 · one hidden layer, trained by backpropagation")
    print("=" * 70)
    net = TinyNet(hidden=2, seed=3)

    a, n_ = gradient_check(net, X, Y)
    print(f"  gradient check on W2[0,0]:  analytic {a:+.8f}   numeric {n_:+.8f}")
    print(f"  absolute difference: {abs(a - n_):.2e}   (should be ~1e-9)\n")

    hist = net.fit(X, Y, lr=0.5, epochs=8000)
    out = net.forward(X)
    print("\n  input      target   predicted")
    for xi, t, p in zip(X, Y.ravel(), out.ravel()):
        print(f"  {xi}   {t:.0f}        {p:.4f}")
    correct = int(((out > 0.5).astype(float) == Y).all(axis=1).sum())
    print(f"\n  {correct} of 4 correct. XOR is solved.")

    print("\n  What the hidden layer learned (its activations per input):")
    for xi, h in zip(X, net.a1):
        print(f"  {xi} -> [{h[0]:.3f}, {h[1]:.3f}]")
    print("\n  The hidden layer RE-REPRESENTS the four points so that they")
    print("  become linearly separable. That re-representation, not the extra")
    print("  parameters, is what a hidden layer buys you.")
