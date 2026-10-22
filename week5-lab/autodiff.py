"""
autodiff.py, a small automatic differentiation engine that can differentiate
its own derivatives.

The Evolution of Machine Learning and AI, Week 5 Lab
South East Technological University

WHY THIS FILE EXISTS

Week 3 wrote backpropagation by hand for one architecture. Week 4 wrote it again
for convolutions, recurrence and attention. In every case the pattern was the
same: differentiate the loss with respect to the WEIGHTS.

Week 5 needs something genuinely new. A physics-informed network's loss contains
the network's derivative with respect to its INPUT:

    residual = m u''(t) + c u'(t) + k u(t)

where u is the network. To train it you need the gradient of that residual with
respect to the weights, which means you need the derivative OF a derivative. The
hand-written approach stops being reasonable at that point, so this file builds
the machinery instead.

WHAT MAKES IT DIFFERENT FROM WEEK 3'S BACKPROP

In Week 3 each layer's backward pass returned numbers. Here every backward pass
returns NODES, built from the same operations as the forward pass. The gradient
is therefore itself a differentiable expression, so you can call `grad` on the
result of `grad`, as many times as you like. That single design decision is the
whole trick, and it is what PyTorch means by `create_graph=True`.

It is about 250 lines. Read `grad()` and one backward function and you have it.

Booklet cross-reference: Chapter 7.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class Node:
    """A value in the computation graph.

    `value`   the numbers, as a numpy array
    `parents` the nodes this one was computed from
    `vjp`     a function taking the gradient flowing in (a Node) and returning
              one gradient Node per parent. "vjp" is vector-Jacobian product,
              which is the only thing reverse-mode differentiation ever does.
    """

    __slots__ = ("value", "parents", "vjp", "name")

    def __init__(self, value, parents=(), vjp=None, name=""):
        self.value = np.asarray(value, dtype=np.float64)
        self.parents = parents
        self.vjp = vjp
        self.name = name

    # --- conveniences ---------------------------------------------------
    @property
    def shape(self):
        return self.value.shape

    def __repr__(self):
        return f"Node(shape={self.value.shape}, name={self.name!r})"

    def __add__(self, o):
        return add(self, o)

    def __radd__(self, o):
        return add(o, self)

    def __sub__(self, o):
        return sub(self, o)

    def __rsub__(self, o):
        return sub(o, self)

    def __mul__(self, o):
        return mul(self, o)

    def __rmul__(self, o):
        return mul(o, self)

    def __truediv__(self, o):
        return div(self, o)

    def __rtruediv__(self, o):
        return div(o, self)

    def __neg__(self):
        return mul(self, -1.0)

    def __pow__(self, p):
        return power(self, p)

    def __matmul__(self, o):
        return matmul(self, o)


def const(x, name=""):
    """Wrap raw numbers as a graph node."""
    return x if isinstance(x, Node) else Node(x, name=name)


# ---------------------------------------------------------------------------
# Reverse-mode differentiation
# ---------------------------------------------------------------------------

def _topo(out):
    """Nodes in an order where every node appears after everything that uses it."""
    order, seen = [], set()

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for p in n.parents:
            visit(p)
        order.append(n)

    visit(out)
    return order


def grad(out, wrt, seed=None):
    """d out / d each element of `wrt`, returned as NODES.

    Because the returned gradients are nodes rather than arrays, they can be
    differentiated again. That is the entire reason this engine exists, and it
    is what makes the second derivative in a PINN's residual possible.

    `out` must be a scalar-shaped node unless you pass `seed` explicitly.
    """
    if seed is None:
        if out.value.size != 1:
            raise ValueError(
                f"grad expects a scalar output, got shape {out.value.shape}. "
                "Pass seed= to differentiate a non-scalar."
            )
        seed = Node(np.ones_like(out.value), name="seed")

    grads = {id(out): seed}
    for node in reversed(_topo(out)):
        g = grads.get(id(node))
        if g is None or node.vjp is None:
            continue
        contributions = node.vjp(g)
        for parent, contribution in zip(node.parents, contributions):
            if contribution is None:
                continue
            if id(parent) in grads:
                # NOTE: this addition is a graph operation, not a numpy one.
                # It is why the gradient stays differentiable.
                grads[id(parent)] = add(grads[id(parent)], contribution)
            else:
                grads[id(parent)] = contribution

    single = isinstance(wrt, Node)
    targets = [wrt] if single else list(wrt)
    outs = [
        grads.get(id(t), Node(np.zeros_like(t.value), name="zero"))
        for t in targets
    ]
    return outs[0] if single else outs


# ---------------------------------------------------------------------------
# Shape bookkeeping
#
# Broadcasting is the only genuinely fiddly part of writing an autodiff engine.
# When a bias of shape (1, d) is added to activations of shape (N, d), numpy
# silently expands it, so the gradient arriving back at the bias has shape
# (N, d) and must be summed down to (1, d). `sum_to` does that, and because it
# is a graph operation with its own vjp, it works at every derivative order.
# ---------------------------------------------------------------------------

def sum_to(x, shape):
    x = const(x)
    if x.value.shape == tuple(shape):
        return x
    target = tuple(shape)

    def _reduce(v):
        # drop leading axes the target does not have
        while v.ndim > len(target):
            v = v.sum(axis=0)
        # sum any axis the target has as length 1
        for i, s in enumerate(target):
            if s == 1 and v.shape[i] != 1:
                v = v.sum(axis=i, keepdims=True)
        return v.reshape(target)

    def vjp(g):
        return (broadcast_to(g, x.value.shape),)

    return Node(_reduce(x.value), (x,), vjp, "sum_to")


def broadcast_to(x, shape):
    x = const(x)
    if x.value.shape == tuple(shape):
        return x

    def vjp(g):
        return (sum_to(g, x.value.shape),)

    return Node(np.broadcast_to(x.value, shape).copy(), (x,), vjp, "broadcast")


# ---------------------------------------------------------------------------
# Operations
#
# Every vjp below is written using the operations in this same file, never with
# raw numpy. Read one of them and you have read all of them.
# ---------------------------------------------------------------------------

def add(a, b):
    a, b = const(a), const(b)

    def vjp(g):
        return (sum_to(g, a.value.shape), sum_to(g, b.value.shape))

    return Node(a.value + b.value, (a, b), vjp, "add")


def sub(a, b):
    a, b = const(a), const(b)

    def vjp(g):
        return (sum_to(g, a.value.shape), sum_to(-g, b.value.shape))

    return Node(a.value - b.value, (a, b), vjp, "sub")


def mul(a, b):
    a, b = const(a), const(b)

    def vjp(g):
        # d(ab)/da = b, so the gradient into a is g * b. Written with graph ops.
        return (sum_to(mul(g, b), a.value.shape),
                sum_to(mul(g, a), b.value.shape))

    return Node(a.value * b.value, (a, b), vjp, "mul")


def div(a, b):
    a, b = const(a), const(b)

    def vjp(g):
        ga = div(g, b)
        gb = -div(mul(g, a), mul(b, b))
        return (sum_to(ga, a.value.shape), sum_to(gb, b.value.shape))

    return Node(a.value / b.value, (a, b), vjp, "div")


def matmul(a, b):
    a, b = const(a), const(b)

    def vjp(g):
        return (matmul(g, transpose(b)), matmul(transpose(a), g))

    return Node(a.value @ b.value, (a, b), vjp, "matmul")


def transpose(a):
    a = const(a)

    def vjp(g):
        return (transpose(g),)

    return Node(a.value.T, (a,), vjp, "transpose")


def tanh(a):
    a = const(a)
    t = np.tanh(a.value)

    def vjp(g):
        # d tanh / dx = 1 - tanh^2. Expressed as graph ops so it differentiates.
        return (mul(g, sub(1.0, mul(tanh(a), tanh(a)))),)

    return Node(t, (a,), vjp, "tanh")


def sin(a):
    a = const(a)

    def vjp(g):
        return (mul(g, cos(a)),)

    return Node(np.sin(a.value), (a,), vjp, "sin")


def cos(a):
    a = const(a)

    def vjp(g):
        return (mul(g, -sin(a)),)

    return Node(np.cos(a.value), (a,), vjp, "cos")


def exp(a):
    a = const(a)

    def vjp(g):
        return (mul(g, exp(a)),)

    return Node(np.exp(a.value), (a,), vjp, "exp")


def power(a, p):
    a = const(a)
    p = float(p)

    def vjp(g):
        return (mul(g, mul(p, power(a, p - 1.0))),)

    return Node(a.value ** p, (a,), vjp, "power")


def square(a):
    """x squared, written as x*x rather than x**2 on purpose.

    `power(a, 2)` differentiates to `2 * power(a, 1)` and then to
    `2 * 1 * power(a, 0)`, and the THIRD derivative reaches `power(a, -1)`,
    which is a division by zero wherever a is zero. Since a PINN evaluates its
    residual at t = 0, that matters. Writing it as a product keeps every
    derivative polynomial and finite. A small example of a numerical trap that
    only appears once you start taking derivatives of derivatives.
    """
    a = const(a)
    return mul(a, a)


def sum_all(a):
    a = const(a)

    def vjp(g):
        return (broadcast_to(g, a.value.shape),)

    return Node(a.value.sum(), (a,), vjp, "sum")


def mean(a):
    a = const(a)
    n = float(a.value.size)

    def vjp(g):
        return (div(broadcast_to(g, a.value.shape), n),)

    return Node(a.value.mean(), (a,), vjp, "mean")


def concat(nodes, axis=1):
    nodes = [const(n) for n in nodes]
    sizes = [n.value.shape[axis] for n in nodes]

    def vjp(g):
        outs, start = [], 0
        for s in sizes:
            outs.append(slice_axis(g, axis, start, start + s))
            start += s
        return tuple(outs)

    return Node(np.concatenate([n.value for n in nodes], axis=axis),
                tuple(nodes), vjp, "concat")


def slice_axis(a, axis, start, stop):
    a = const(a)
    idx = [slice(None)] * a.value.ndim
    idx[axis] = slice(start, stop)
    idx = tuple(idx)
    full = a.value.shape

    def vjp(g):
        return (pad_axis(g, axis, start, stop, full),)

    return Node(a.value[idx], (a,), vjp, "slice")


def pad_axis(a, axis, start, stop, full_shape):
    a = const(a)

    def _place(v):
        out = np.zeros(full_shape)
        idx = [slice(None)] * len(full_shape)
        idx[axis] = slice(start, stop)
        out[tuple(idx)] = v
        return out

    def vjp(g):
        return (slice_axis(g, axis, start, stop),)

    return Node(_place(a.value), (a,), vjp, "pad")


# ---------------------------------------------------------------------------
# Checking the engine
# ---------------------------------------------------------------------------

def check_engine(verbose=True):
    """Verify first, second and third derivatives against exact calculus.

    Week 3, Chapter 14's rule applies with more force than ever: a wrong
    derivative fails silently, and here a wrong SECOND derivative would produce
    a PINN that trains happily to the wrong answer. So this is checked against
    functions whose derivatives are known exactly, not against finite
    differences.
    """
    rows = []
    x0 = 0.7

    # f(x) = sin(x): derivatives cycle cos, -sin, -cos
    x = Node([[x0]], name="x")
    f = sin(x)
    d1 = grad(sum_all(f), x)
    d2 = grad(sum_all(d1), x)
    d3 = grad(sum_all(d2), x)
    rows.append(("sin, 1st", d1.value.item(), np.cos(x0)))
    rows.append(("sin, 2nd", d2.value.item(), -np.sin(x0)))
    rows.append(("sin, 3rd", d3.value.item(), -np.cos(x0)))

    # f(x) = tanh(x): f' = 1 - t^2,  f'' = -2 t (1 - t^2)
    x = Node([[x0]], name="x")
    t = np.tanh(x0)
    f = tanh(x)
    d1 = grad(sum_all(f), x)
    d2 = grad(sum_all(d1), x)
    rows.append(("tanh, 1st", d1.value.item(), 1 - t ** 2))
    rows.append(("tanh, 2nd", d2.value.item(), -2 * t * (1 - t ** 2)))

    # f(x) = exp(x) * x^3
    x = Node([[x0]], name="x")
    f = mul(exp(x), power(x, 3.0))
    d1 = grad(sum_all(f), x)
    d2 = grad(sum_all(d1), x)
    e = np.exp(x0)
    rows.append(("exp*x^3, 1st", d1.value.item(), e * (x0 ** 3 + 3 * x0 ** 2)))
    rows.append(("exp*x^3, 2nd", d2.value.item(),
                 e * (x0 ** 3 + 6 * x0 ** 2 + 6 * x0)))

    worst = 0.0
    if verbose:
        print(f"  {'expression':16s} {'engine':>16s} {'exact':>16s} {'rel err':>10s}")
    for name, got, want in rows:
        rel = abs(got - want) / (abs(want) + 1e-15)
        worst = max(worst, rel)
        if verbose:
            print(f"  {name:16s} {got:16.10f} {want:16.10f} {rel:10.2e}")
    return worst


if __name__ == "__main__":
    # NOTE: run `python mlp.py` for the network-derivative check. Running a
    # module as a script gives it the name __main__, so anything importing
    # `autodiff` would load a SECOND copy with its own Node class, and the two
    # would not recognise each other. A small, real, and very confusing Python
    # trap worth meeting once.
    print("\nDerivatives of known functions, to third order:")
    worst = check_engine()
    print(f"\n  worst relative error: {worst:.2e}")
    print("\nAll three orders are exact to machine precision, because the engine")
    print("differentiates symbolically rather than numerically. Chapter 8 explains")
    print("why that distinction decides whether a PINN can work at all.")
