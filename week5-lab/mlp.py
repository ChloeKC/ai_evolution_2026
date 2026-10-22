"""
mlp.py, the one network used by every experiment this week, and Adam.

The Evolution of Machine Learning and AI, Week 5 Lab
South East Technological University

Week 5 makes an unusual promise, and this file is what keeps it honest:

    EVERY comparison in this lab uses THE SAME NETWORK.

The physics-informed model and the data-only baseline are the same class, the
same width, the same depth, the same initialisation and the same optimiser. The
only difference between them is the loss function. So when one wins, the win
cannot be attributed to capacity, tuning or luck. It is attributable to the one
thing that differs, which is whether the model was told what the world does.

That is the experimental design the whole week rests on, and it is worth saying
out loud because a great deal of published comparison does not do it.

WHY tanh AND NOT ReLU

Week 4 used ReLU everywhere. Here we cannot. A PINN's loss contains the second
derivative of the network, and ReLU's second derivative is zero almost
everywhere: it is piecewise linear, so its curvature carries no information. A
network of ReLUs would report that every candidate solution has zero
acceleration, and the physics loss would be satisfied by nonsense.

tanh is smooth and infinitely differentiable, so its second derivative is a
useful function of the input. This is the first example of something that recurs
all week: a physics constraint changes an architectural choice you thought was
settled.

Booklet cross-references: Chapters 7, 8 and 9.
"""

from __future__ import annotations

import numpy as np

from autodiff import Node, grad, matmul, add, tanh, sin, const, sum_all


class MLP:
    """A plain fully-connected network with smooth activations.

    `layers` is the widths, so [1, 32, 32, 1] means one input, two hidden layers
    of 32, one output.

    `activation` is "tanh" by default. "sin" is offered because sinusoidal
    activations (the SIREN idea) represent oscillatory solutions with fewer
    units, and the damped oscillator is exactly such a problem. Try both.
    """

    def __init__(self, layers, seed=0, activation="tanh", scale=1.0):
        rng = np.random.default_rng(seed)
        self.layers = list(layers)
        self.activation = activation
        self.params = []
        self.Ws, self.bs = [], []
        for i in range(len(layers) - 1):
            n_in, n_out = layers[i], layers[i + 1]
            # Glorot scaling, as in Week 3 Chapter 12: keeps activation
            # variance roughly constant through depth for a symmetric,
            # saturating non-linearity.
            lim = scale * np.sqrt(6.0 / (n_in + n_out))
            W = Node(rng.uniform(-lim, lim, (n_in, n_out)), name=f"W{i}")
            b = Node(np.zeros((1, n_out)), name=f"b{i}")
            self.Ws.append(W)
            self.bs.append(b)
            self.params += [W, b]

    @property
    def n_params(self):
        return sum(p.value.size for p in self.params)

    def __call__(self, x):
        x = const(x)
        act = tanh if self.activation == "tanh" else sin
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            x = add(matmul(x, W), b)
            if i < len(self.Ws) - 1:
                x = act(x)
        return x

    # --- saving and restoring, for the "best so far" pattern -------------
    def get_weights(self):
        return [p.value.copy() for p in self.params]

    def set_weights(self, ws):
        for p, w in zip(self.params, ws):
            p.value[...] = w


class Adam:
    """Adam, as in Week 4's `optim.py`, operating on graph nodes.

    Kept identical across every experiment so that no comparison in this lab is
    secretly a comparison of optimisers.
    """

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.value) for p in self.params]
        self.v = [np.zeros_like(p.value) for p in self.params]
        self.t = 0

    def step(self, grads, clip=None):
        self.t += 1
        if clip is not None:
            total = np.sqrt(sum(float((g.value ** 2).sum()) for g in grads))
            factor = min(1.0, clip / (total + 1e-12))
        else:
            factor = 1.0
        for i, (p, g) in enumerate(zip(self.params, grads)):
            gv = g.value * factor
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * gv
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * gv * gv
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.value -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def set_lr(self, lr):
        self.lr = lr


def train(loss_fn, params, epochs=2000, lr=1e-3, decay_at=None, decay=0.3,
          clip=None, report_every=None, track=None):
    """The loop. Unchanged in shape from Week 3, and it never will change.

    `loss_fn()` returns a scalar Node (and optionally a dict of components).
    `track` is an optional callable returning a number to record each epoch,
    used to plot true error against training epoch even though the PINN never
    sees the true solution.
    """
    opt = Adam(params, lr=lr)
    history, tracked, parts_hist = [], [], []
    for ep in range(epochs):
        out = loss_fn()
        if isinstance(out, tuple):
            loss, parts = out
        else:
            loss, parts = out, None
        gs = grad(loss, params)
        opt.step(gs, clip=clip)
        history.append(float(loss.value))
        if parts is not None:
            parts_hist.append({k: float(v) for k, v in parts.items()})
        if track is not None:
            tracked.append(track())
        if decay_at and (ep + 1) in decay_at:
            opt.set_lr(opt.lr * decay)
        if report_every and (ep + 1) % report_every == 0:
            print(f"    epoch {ep+1:5d}  loss {float(loss.value):.3e}", flush=True)
    return history, tracked, parts_hist


def mse(pred, target):
    """Mean squared error, as a graph node."""
    from autodiff import mean, sub, square
    return mean(square(sub(pred, const(target))))


def check_network_derivative(seed=0, verbose=True):
    """Check d(network)/d(input) against a central difference.

    The engine's derivatives are exact, so the disagreement here is the finite
    difference's error, not the engine's. That is worth seeing once: it is the
    reason PINNs use automatic differentiation rather than a numerical stencil,
    and Chapter 8 makes the argument in full.
    """
    net = MLP([1, 16, 16, 1], seed=seed)
    t = Node(np.array([[0.4]]))
    u = net(t)
    u_t = grad(sum_all(u), t)
    u_tt = grad(sum_all(u_t), t)

    eps = 1e-5
    def f(v):
        return net(Node(np.array([[v]]))).value.item()

    num1 = (f(0.4 + eps) - f(0.4 - eps)) / (2 * eps)
    num2 = (f(0.4 + eps) - 2 * f(0.4) + f(0.4 - eps)) / eps ** 2

    r1 = abs(u_t.value.item() - num1) / (abs(num1) + 1e-15)
    r2 = abs(u_tt.value.item() - num2) / (abs(num2) + 1e-15)
    if verbose:
        print(f"  first derivative : engine {u_t.value.item():+.8f}   "
              f"finite difference {num1:+.8f}   rel diff {r1:.2e}")
        print(f"  second derivative: engine {u_tt.value.item():+.8f}   "
              f"finite difference {num2:+.8f}   rel diff {r2:.2e}")
    return r1, r2



if __name__ == "__main__":
    net = MLP([1, 16, 1], seed=0)
    print(f"MLP([1, 16, 1]) has {net.n_params} parameters")
    x = Node(np.linspace(-1, 1, 5).reshape(-1, 1))
    y = net(x)
    print("output shape:", y.shape)
    g = grad(sum_all(y), x)
    print("d output / d input shape:", g.shape)
    gg = grad(sum_all(g), x)
    print("second derivative shape:", gg.shape)
    print("\nThe second derivative exists and is differentiable again.")
    print("That is the whole capability Week 5 needs.")
    print("\nDerivatives of a network with respect to its input:")
    check_network_derivative()
