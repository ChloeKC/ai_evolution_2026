"""lecture_steps.py: one helper per lecture part, so each lab exercise is ONE cell
and prints exactly what the matching slides showed.

    Exercise 1 (Part 1, slides 6 to 11)   house_dials(), watch_the_loop()
    Exercise 2 (Part 2, slides 12 to 19)  four_shapes()
    Exercise 3 (Part 3, slides 20 to 23)  tree_depth_sweep(), the_gap()
    Exercise 4 (Part 4, slides 24 to 30)  walk(), step_sizes(), bumpy_hill()
    Exercise 5 (Part 5, slides 31 to 37)  perceptron_vs_xor(), hidden_layer(), no_bend()
    Exercise 6 (Part 6, slides 38 to 43)  blame_by_hand(), five_lines()

You never edit this file. Every function prints a labelled block (Printout A, Table B ...)
that the questions in SUBMISSION.md point at by name.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

# --------------------------------------------------------------------------- Part 1
HOUSES = pd.DataFrame({"size_m2": [50, 70, 90, 110, 130], "price_k": [155, 205, 260, 305, 360]})


def house_dials(a: float, b: float) -> pd.DataFrame:
    """Slide 7: price = a x size + b. Set the two dials, see every guess and the loss."""
    df = HOUSES.copy()
    df["guess_k"] = a * df.size_m2 + b
    df["miss_k"] = df.guess_k - df.price_k
    loss = float((df.miss_k ** 2).mean())
    print(f"Printout A: dials a = {a}, b = {b}   ->   loss (mean squared miss) = {loss:,.1f}")
    return df.round(1)


def watch_the_loop(a: float = 0.0, b: float = 0.0, lr: float = 0.00003, steps: int = 6, show: int = 6) -> pd.DataFrame:
    """Slide 8: Guess, Check, Blame, Nudge, printed once per trip round the loop."""
    x = HOUSES.size_m2.to_numpy(float); y = HOUSES.price_k.to_numpy(float)
    rows = []
    for t in range(steps):
        guess = a * x + b                       # 1 GUESS   (forward)
        miss = guess - y
        loss = float((miss ** 2).mean())         # 2 CHECK   (loss)
        blame_a = float((2 * miss * x).mean())   # 3 BLAME   (backward): how the loss changes if a moves
        blame_b = float((2 * miss).mean())
        rows.append({"trip": t, "a": a, "b": b, "loss": loss, "blame on a": blame_a, "blame on b": blame_b})
        a = a - lr * blame_a                     # 4 NUDGE   (update): new dial = old dial - step size x slope
        b = b - lr * blame_b
    df = pd.DataFrame(rows)
    print(f"Table B: the loop, {steps} trips, step size {lr}. Columns are the four boxes: guess and check give 'loss', blame gives the two 'blame' columns, nudge changes a and b for the next row.")
    return df.head(show).round(3)


def keep_going(steps: int = 20000, lr: float = 0.00003) -> None:
    """Slide 8, build 5: 'then go round again, thousands of times'."""
    x = HOUSES.size_m2.to_numpy(float); y = HOUSES.price_k.to_numpy(float)
    a = b = 0.0
    for t in range(steps):
        miss = a * x + b - y
        a -= lr * float((2 * miss * x).mean()); b -= lr * float((2 * miss).mean())
    loss = float(((a * x + b - y) ** 2).mean())
    print(f"Printout C: after {steps} trips round the loop, a = {a:.3f}, b = {b:.3f}, loss = {loss:.1f}")
    print(f"            read as a sentence: each extra square metre adds about {a*1000:,.0f} euro.")


# --------------------------------------------------------------------------- Part 2
def four_shapes() -> pd.DataFrame:
    """Slides 13 to 18: seven models, one table, with the shape each one draws."""
    import classical
    t = classical.compare()
    shape = {
        "Logistic regression": "one straight line", "SVM (linear)": "one straight line",
        "Decision tree (depth 3)": "right-angled steps", "Decision tree (unlimited)": "right-angled steps",
        "Random forest": "many staircases averaged", "SVM (RBF kernel)": "a curve (bent page)",
        "Neural net (8, 8)": "many small bends",
    }
    t.insert(1, "shape it draws", t.model.map(shape))
    print("Table A: seven models trained on the same dots. train = score on dots it studied, test = on dots it never saw, gap = train minus test.")
    return t.drop(columns=["assumption encoded"]).round(3)


# --------------------------------------------------------------------------- Part 3
def tree_depth_sweep(depths=(1, 2, 3, 5, 8, None), seed: int = 0) -> pd.DataFrame:
    """Slides 15 and 22: let the tree keep asking questions and watch the gap open."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    import classical
    Xtr, Xte, ytr, yte = classical.load(seed=seed)[:4]
    rows = []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=seed).fit(Xtr, ytr)
        tr, te = m.score(Xtr, ytr), m.score(Xte, yte)
        rows.append({"max questions (depth)": "unlimited" if d is None else d, "leaves": m.get_n_leaves(),
                     "train": tr, "test": te, "gap": tr - te})
    print("Table A: the same tree allowed more and more questions. Watch train rise to 1.000 while test stops improving and the gap opens.")
    return pd.DataFrame(rows).round(3)


def the_gap() -> None:
    """Slide 22: the two rows that matter."""
    import classical
    t = classical.compare().set_index("model")
    small, big = t.loc["Decision tree (depth 3)"], t.loc["Decision tree (unlimited)"]
    print("Printout B: the diagnosis from slide 22")
    print(f"  depth-3 tree      train {small.train:.3f}  test {small.test:.3f}  gap {small.gap:.3f}")
    print(f"  unlimited tree    train {big.train:.3f}  test {big.test:.3f}  gap {big.gap:.3f}   <- perfect on training, worse on new dots: memorised")


# --------------------------------------------------------------------------- Part 4
def walk(x0: float = 4.2, lr: float = 0.25, steps: int = 8) -> pd.DataFrame:
    """Slide 25: feel the slope, step down, repeat. One row per step on the bowl loss = x^2 / 2."""
    import gradient_descent as gd
    r = gd.descend(gd.bowl, gd.bowl_d, x0, lr=lr, steps=steps)
    df = pd.DataFrame({"step": range(len(r.xs)), "dial (x)": r.xs, "loss (height)": r.losses,
                       "slope under your feet": list(r.grads) + [np.nan]})
    df["step length"] = df["dial (x)"].diff().abs().shift(-1)
    print(f"Table A: the walker, starting at {x0}, step size {lr}. Each step = step size x slope. Watch the steps shrink on their own as the slope flattens.")
    return df.round(3)


def step_sizes(rates=(0.02, 0.1, 0.9, 1.98, 2.05)) -> pd.DataFrame:
    """Slides 27 and 28: same hill, same start, five step sizes."""
    import gradient_descent as gd
    _, sweep = gd.lr_sweep(rates=rates)
    print("Table B: five learning rates from x = 4, forty steps. Read the verdict column.")
    return sweep[["lr", "final_x", "final_loss", "verdict"]].round(4)


def try_lr(lr: float, steps: int = 40) -> None:
    """Your own learning rate."""
    import gradient_descent as gd
    r = gd.descend(gd.bowl, gd.bowl_d, 4.0, lr=lr, steps=steps)
    final = float("inf") if r.diverged else float(r.losses[-1])
    print(f"learning rate {lr:<6}  final loss {final:<12.4g}  ->  {gd._verdict(r, lr)}")


def bumpy_hill() -> pd.DataFrame:
    """Slide 29: not every hill is a bowl."""
    import gradient_descent as gd
    _, dips = gd.local_minima_demo()
    print("Table C: same step size, three starting points, a hill with two dips. One walker stops in the wrong dip.")
    return dips.round(4)


# --------------------------------------------------------------------------- Part 5
XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
XOR_Y = np.array([0, 1, 1, 0], float)


def perceptron_vs_xor(epochs: int = 200) -> pd.DataFrame:
    """Slides 32 and 33: one neuron meets the four dots."""
    import scratch_net as sn
    res = sn.perceptron(XOR_X, XOR_Y.astype(int), epochs=epochs)
    w, b = res[0], res[1]
    pred = ((XOR_X @ w + b) > 0).astype(int)
    df = pd.DataFrame({"input": [tuple(map(int, r)) for r in XOR_X], "answer": XOR_Y.astype(int), "perceptron says": pred})
    df["right?"] = np.where(df.answer == df["perceptron says"], "yes", "NO")
    print(f"Table A: the perceptron after {epochs} passes over the four dots. Best it ever manages: {int((df.answer == df['perceptron says']).sum())} of 4.")
    return df


def hidden_layer(hidden: int = 2, epochs: int = 4000, seed: int = 3):
    """Slide 34: put two neurons in between, and watch the dots move."""
    import scratch_net as sn
    net = sn.TinyNet(hidden=hidden, seed=seed)
    hist = net.fit(XOR_X, XOR_Y.reshape(-1, 1), lr=0.5, epochs=epochs, log_every=epochs)
    out = net.forward(XOR_X); h = net.a1
    df = pd.DataFrame({"input": [tuple(map(int, r)) for r in XOR_X], "answer": XOR_Y.astype(int),
                       "network says": np.round(np.ravel(out), 2), "hidden layer moved it to": [tuple(np.round(r, 2)) for r in h]})
    correct = int((np.round(np.ravel(out)) == XOR_Y).sum())
    print(f"Table B: {hidden} hidden neurons, {epochs} trips round the loop. Final loss {hist[-1]:.4f}. {correct} of 4 correct.")
    print("         Look at the last column: the two dots whose answer is 1 now sit on top of each other.")
    return df, hist


def no_bend(epochs: int = 4000, seed: int = 3) -> pd.DataFrame:
    """Slide 35: two hidden neurons but no activation. A line after a line is still a line."""
    rng = np.random.default_rng(seed)
    W1, b1 = rng.normal(0, 0.5, (2, 2)), np.zeros(2); W2, b2 = rng.normal(0, 0.5, (2, 1)), np.zeros(1)
    X, Y = XOR_X, XOR_Y.reshape(-1, 1)
    for _ in range(epochs):
        h = X @ W1 + b1; out = h @ W2 + b2          # no bend anywhere
        d_out = 2 * (out - Y) / len(X); d_h = d_out @ W2.T
        W2 -= 0.05 * h.T @ d_out; b2 -= 0.05 * d_out.sum(0); W1 -= 0.05 * X.T @ d_h; b1 -= 0.05 * d_h.sum(0)
    out = (X @ W1 + b1) @ W2 + b2
    df = pd.DataFrame({"input": [tuple(map(int, r)) for r in X], "answer": XOR_Y.astype(int), "network says": np.round(np.ravel(out), 2)})
    correct = int((np.round(np.ravel(out)) == XOR_Y).sum())
    print(f"Table C: same two hidden neurons with the bend switched OFF, {epochs} trips. {correct} of 4 correct. It outputs about 0.5 for everything: the best a straight line can do.")
    return df


# --------------------------------------------------------------------------- Part 6
def blame_by_hand() -> None:
    """Slide 39: the smallest network, forward then backward, every number printed."""
    from scratch_net import sigmoid
    x, target = 1.0, 1.0
    w1, w2 = 0.60, 0.80                              # the two dials
    h_in = w1 * x;  h = sigmoid(h_in)                # hidden neuron: multiply, then bend
    o_in = w2 * h;  o = sigmoid(o_in)                # output neuron
    loss = 0.5 * (o - target) ** 2
    print("Printout A: FORWARD (numbers go left to right)")
    print(f"  input x = {x}   ->  hidden = sigmoid({w1} x {x}) = {h:.3f}   ->  output = sigmoid({w2} x {h:.3f}) = {o:.3f}")
    print(f"  CHECK: target {target}, loss = 0.5 x ({o:.3f} - {target})^2 = {loss:.3f}")
    d_o = (o - target) * o * (1 - o)                 # blame arriving at the output neuron
    blame_w2 = d_o * h
    d_h = d_o * w2 * h * (1 - h)                     # blame passed back to the hidden neuron
    blame_w1 = d_h * x
    print("Printout B: BLAME (walk backwards, output first)")
    print(f"  share of blame at the output neuron  = {d_o:+.4f}")
    print(f"  blame on w2 (output dial)            = {blame_w2:+.4f}   <- slope of the loss with respect to w2")
    print(f"  share passed back to the hidden neuron = {d_h:+.4f}")
    print(f"  blame on w1 (hidden dial)            = {blame_w1:+.4f}   <- slope of the loss with respect to w1")
    eps = 1e-6
    def L(a, b):
        return 0.5 * (sigmoid(b * sigmoid(a * x)) - target) ** 2
    print("Printout C: CHECK THE BLAME by nudging each dial a hair and re-running forward (the gradient check)")
    print(f"  w2: ({L(w1, w2 + eps):.7f} - {L(w1, w2 - eps):.7f}) / {2*eps:g} = {(L(w1, w2+eps)-L(w1, w2-eps))/(2*eps):+.4f}   matches")
    print(f"  w1: ({L(w1 + eps, w2):.7f} - {L(w1 - eps, w2):.7f}) / {2*eps:g} = {(L(w1+eps, w2)-L(w1-eps, w2))/(2*eps):+.4f}   matches")
    lr = 1.0
    print("Printout D: NUDGE, step size 1.0: new dial = old dial - step size x slope")
    print(f"  w1: {w1} - {lr} x ({blame_w1:+.4f}) = {w1 - lr*blame_w1:.4f}     w2: {w2} - {lr} x ({blame_w2:+.4f}) = {w2 - lr*blame_w2:.4f}")
    o2 = sigmoid((w2 - lr*blame_w2) * sigmoid((w1 - lr*blame_w1) * x))
    print(f"  forward again with the nudged dials: output {o2:.3f} (was {o:.3f}), loss {0.5*(o2-target)**2:.3f} (was {loss:.3f}). Smaller. That is one trip round the loop.")


def five_lines() -> None:
    """Slide 42: the five PyTorch lines, mapped onto the loop; runs if torch is installed."""
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed, which is fine this week. The five lines, and the loop step each one is:")
        print("  optimiser.zero_grad()      # 0. wipe last time's blame\n  pred = model(X)            # 1. GUESS   (forward)\n  loss = loss_fn(pred, Y)    # 2. CHECK   (loss)\n  loss.backward()            # 3. BLAME   (backward)\n  optimiser.step()           # 4. NUDGE   (update)")
        return
    torch.manual_seed(0)
    X = torch.tensor(XOR_X, dtype=torch.float32); Y = torch.tensor(XOR_Y, dtype=torch.float32).reshape(-1, 1)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sigmoid(), torch.nn.Linear(2, 1), torch.nn.Sigmoid())
    optimiser = torch.optim.SGD(model.parameters(), lr=0.5); loss_fn = torch.nn.MSELoss()
    for epoch in range(4000):
        optimiser.zero_grad()      # 0. wipe last time's blame
        pred = model(X)            # 1. GUESS   (forward)
        loss = loss_fn(pred, Y)    # 2. CHECK   (loss)
        loss.backward()            # 3. BLAME   (backward)
        optimiser.step()           # 4. NUDGE   (update)
        if epoch % 1000 == 0 or epoch == 3999:
            print(f"  epoch {epoch:4d}  loss {loss.item():.4f}")
    print("  final answers:", [round(v, 2) for v in model(X).detach().ravel().tolist()], "for XOR targets [0, 1, 1, 0]")
