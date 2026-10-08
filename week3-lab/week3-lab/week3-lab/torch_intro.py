"""
torch_intro.py, the same network again, this time in PyTorch.

The Evolution of Machine Learning and AI · Week 3 Lab
South East Technological University

Booklet cross-reference: Chapter 13, "PyTorch, and what it is doing for you".

Read `scratch_net.py` FIRST. This file is deliberately the same network on the
same problem, so that you can put the two side by side and see exactly which of
your hand-written lines each PyTorch call replaced:

    scratch_net.py                       torch_intro.py
    ------------------------------------ ---------------------------------
    self.z1 = X @ self.W1 + self.b1      nn.Linear(2, h)
    sigmoid(z)                           nn.Sigmoid()  /  torch.sigmoid
    mean((a2 - y) ** 2)                  nn.MSELoss()
    the entire backward() method         loss.backward()
    param -= lr * grad                   optimiser.step()
    (nothing: numpy has no state)        optimiser.zero_grad()

The last line is the one that catches everybody. PyTorch ACCUMULATES gradients
by default, so if you forget to zero them you are descending on the sum of every
gradient since the program started. It does not error. It just learns nonsense.
"""

from __future__ import annotations

import numpy as np

# A broken or half-finished torch install raises OSError, not ImportError, so
# catch broadly. Running out of disk part-way through `pip install torch` is a
# genuinely common way to end up with a package that imports and then explodes.
try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
    TORCH_ERROR = None
except Exception as _e:                                # pragma: no cover
    HAVE_TORCH = False
    TORCH_ERROR = f"{type(_e).__name__}: {_e}"


X_np = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32)
Y_np = np.array([[0.], [1.], [1.], [0.]], dtype=np.float32)


def build(hidden: int = 4, seed: int = 0):
    """The whole model. Three lines, and every one has an analogue above."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(2, hidden),
        nn.Sigmoid(),
        nn.Linear(hidden, 1),
        nn.Sigmoid(),
    )


def train(model, epochs=4000, lr=0.5, log_every=1000, momentum=0.0):
    """The canonical PyTorch training loop. Learn these five lines by heart:
    you will write them, in this order, every week from now until Week 12.
    """
    X = torch.tensor(X_np)
    Y = torch.tensor(Y_np)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    history = []
    for epoch in range(epochs + 1):
        optimiser.zero_grad()          # 1. forget the previous step's gradients
        pred = model(X)                # 2. forward:  compute the prediction
        loss = loss_fn(pred, Y)        # 3. measure:  how wrong is it?
        loss.backward()                # 4. backward: assign blame to every weight
        optimiser.step()               # 5. update:   take one step downhill

        history.append(loss.item())
        if log_every and epoch % log_every == 0:
            print(f"    epoch {epoch:>5}   loss {loss.item():.6f}")
    return np.array(history)


def show_autograd():
    """Autograd, on a function small enough to differentiate in your head.

        f(x) = 3x^2 + 2x        f'(x) = 6x + 2        f'(2) = 14
    """
    x = torch.tensor(2.0, requires_grad=True)
    f = 3 * x ** 2 + 2 * x
    f.backward()
    print(f"    f(2)  = {f.item():.1f}")
    print(f"    f'(2) = {x.grad.item():.1f}   (by hand: 6*2 + 2 = 14)")
    print("    PyTorch did not do algebra. It recorded every operation as you")
    print("    performed it, then walked that record backwards applying the")
    print("    chain rule. That record is the 'computational graph'.")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def the_zero_grad_trap(epochs=400):
    """Run the loop correctly, then without zero_grad(). Nothing errors."""
    results = {}
    for label, zero in (("with optimiser.zero_grad()", True),
                        ("WITHOUT zero_grad()", False)):
        model = build(hidden=4, seed=0)
        X, Y = torch.tensor(X_np), torch.tensor(Y_np)
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        for _ in range(epochs):
            if zero:
                opt.zero_grad()
            loss = loss_fn(model(X), Y)
            loss.backward()
            opt.step()
        results[label] = round(float(loss.item()), 6)
    return results


if __name__ == "__main__":
    if not HAVE_TORCH:
        print(f"PyTorch is not usable here  ({TORCH_ERROR})")
        print("  pip install torch          (or: bash setup.sh --full)")
        print("  In Colab it is already there.")
        print("\nEverything this file demonstrates is also in scratch_net.py,")
        print("which needs only numpy. Run that first regardless: it is where")
        print("the derivatives are actually written out.")
        raise SystemExit(0)

    print("=" * 70)
    print("1 · autograd, on something you can check by hand")
    print("=" * 70)
    show_autograd()

    print("\n" + "=" * 70)
    print("2 · the same XOR network, in PyTorch")
    print("=" * 70)
    model = build(hidden=4, seed=0)
    print(model)
    print(f"  trainable parameters: {count_parameters(model)}")
    print()
    train(model)

    with torch.no_grad():
        out = model(torch.tensor(X_np)).numpy()
    print("\n  input      target   predicted")
    for xi, t, p in zip(X_np, Y_np.ravel(), out.ravel()):
        print(f"  {xi}   {t:.0f}        {p:.4f}")

    print("\n" + "=" * 70)
    print("3 · the trap: forgetting zero_grad() does not raise an error")
    print("=" * 70)
    for label, final in the_zero_grad_trap().items():
        print(f"  {label:<30} final loss {final}")
    print("\n  Neither run crashed. One of them learned. This is why the five")
    print("  lines of the training loop are worth memorising in order.")
