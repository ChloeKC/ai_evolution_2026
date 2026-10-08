"""lecture_steps.py: one helper per lecture idea, so each lab exercise is ONE cell
and prints exactly what the matching slides showed.

    Exercise 1 (Part 1)  guess_the_rule(), the_race(), shuffle_test()
    Exercise 2 (Part 2)  by_hand(), kernel_walk(), equivariance()
    Exercise 3 (Part 2)  depth_reaches(), the_head(), no_payoff()
    Exercise 4 (Part 3)  a_power(), gradient_through_time(), half_life(), memory_task()
    Exercise 5 (Part 4)  soft_lookup(), four_instructions(), cannot_see_order(), the_bill()
    Exercise 6 (Part 5)  the_bargain(), seven_questions()

You never edit this file. Every function prints a labelled block (Printout A,
Table B ...) that the questions in SUBMISSION.md point at by name.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# --------------------------------------------------------------------------- Part 1

def guess_the_rule() -> None:
    """Slide: 1, 2, 3 map to 2, 4, 6. What does 4 map to?"""
    rules = {
        "double it": lambda n: 2 * n,
        "double it, unless it is 4, then 1000": lambda n: 1000 if n == 4 else 2 * n,
        "n + n": lambda n: n + n,
        "the n-th even number": lambda n: 2 * n,
        "n squared minus n squared plus 2n, unless n is 4, then 9": lambda n: 9 if n == 4 else 2 * n,
    }
    print("Printout A: five different rules. Every one of them fits the three examples you were given.")
    print(f"  {'rule':<52} 1  2  3   ->  4")
    for name, f in rules.items():
        print(f"  {name:<52} {f(1)}  {f(2)}  {f(3)}   ->  {f(4)}")
    print("\n  The examples cannot choose between them. YOU chose 'double it', because you prefer simple rules.")
    print("  That preference is the inductive bias. Without one, there is no answer to 'what comes next'.")


def the_race(n_train: int = 128, epochs: int = 120) -> pd.DataFrame:
    """Slide: at 128 images, 700 parameters beats 16,708. About 20 seconds."""
    import inductive_bias as ib
    from cnn_numpy import make_cnn, make_mlp
    Xtr, ytr = ib.make_motifs(n_train, seed=0)
    Xte, yte = ib.make_motifs(1000, seed=99)
    rows = []
    for name, builder in (("CNN (assumes: nearby pixels matter, same pattern anywhere)", make_cnn),
                          ("MLP (assumes nothing about space)", make_mlp)):
        net = ib.train(builder, Xtr, ytr, seed=0, epochs=epochs)
        rows.append({"model": name, "parameters": net.n_params() if callable(net.n_params) else net.n_params, "train": round(net.score(Xtr, ytr), 3), "test": round(net.score(Xte, yte), 3)})
    df = pd.DataFrame(rows)
    print(f"Table B: the same {n_train} training images, two models. Chance is 0.250. The MLP has 24 times the parameters and scores at chance on new images.")
    return df


def shuffle_test() -> pd.DataFrame:
    """Slide: destroy adjacency and nothing else. About 30 seconds."""
    import inductive_bias as ib
    df, perm = ib.experiment_shuffle(n_train=512)
    print("Table C: one fixed shuffle of the 256 pixels, applied to every image. Nothing is deleted; only 'which pixels are neighbours' is destroyed.")
    return df.round(3)


# --------------------------------------------------------------------------- Part 2

PATCH = np.array([[0.1, 0.2, 0.9], [0.1, 0.2, 0.9], [0.1, 0.2, 0.9]])
EDGE = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)


def by_hand() -> None:
    """Slide: a convolution by hand. A dark-to-bright patch times a vertical edge detector."""
    print("Printout A: the patch (dark on the left, bright on the right) and the vertical-edge kernel")
    print("  patch:            kernel:")
    for r in range(3):
        print("  " + "  ".join(f"{v:4.1f}" for v in PATCH[r]) + "      " + "  ".join(f"{v:4.0f}" for v in EDGE[r]))
    prods = PATCH * EDGE
    print("\nPrintout B: multiply each pair, row by row, then add everything up")
    for r in range(3):
        print("  " + " + ".join(f"({PATCH[r, c]:.1f} x {EDGE[r, c]:.0f})" for c in range(3)) + f"  =  {prods[r].sum():.1f}")
    print(f"  total = {prods.sum():.1f}     <- a strong response: there IS a left-to-right edge here")
    flat = np.full((3, 3), 0.5)
    print(f"\nPrintout C: the same kernel on a flat patch (every value 0.5): total = {(flat * EDGE).sum():.1f}  (the weights sum to zero, so flat gives nothing)")
    print(f"Printout D: the patch reversed (bright on the left): total = {(PATCH[:, ::-1] * EDGE).sum():.1f}  (the sign says which way round the edge is)")


def kernel_walk() -> pd.DataFrame:
    """Slide: watch the kernel walk. A 6x6 image with one vertical edge, the same nine numbers slid everywhere."""
    from convolution import convolve2d
    img = np.zeros((6, 6)); img[:, 3:] = 1.0
    out = convolve2d(img, EDGE)
    print("Printout A: a 6 x 6 image, dark on the left, bright on the right:")
    print(pd.DataFrame(img).astype(int).to_string(index=False, header=False))
    print("\nTable B: the feature map. The SAME nine numbers were used at every position; each output is nine multiplies and eight adds.")
    print("        Big numbers mark WHERE the edge is. Zeros everywhere the patch is flat.")
    return pd.DataFrame(out).astype(int)


def equivariance() -> pd.DataFrame:
    """Slide: move the input and the output moves. Exactly zero error for convolution."""
    from convolution import equivariance_test
    df = equivariance_test()
    print("Table C: shift the image, then convolve; convolve, then shift the result. For a convolution the two are identical. A dense layer treats the shifted image as something new.")
    return df


# --------------------------------------------------------------------------- Part 2, second half

def depth_reaches() -> pd.DataFrame:
    """Slide: depth turns local into global. The receptive field."""
    from convolution import receptive_field_table
    df = receptive_field_table()
    print("Table A: each layer sees only a 3 x 3 window, but stacked layers see further and further. By the last layer one unit depends on the whole image.")
    return df


def the_head() -> pd.DataFrame:
    """Slide: equivariance is not invariance. Same convolutions, two different endings. About 35 seconds."""
    import inductive_bias as ib
    df = ib.experiment_unseen_positions()
    print("Table B: trained with the pattern only on the LEFT of the image, tested with it only on the RIGHT. Same convolutional layers; the only difference is how the network ends.")
    return df.round(3)


def no_payoff() -> pd.DataFrame:
    """Slide: where a CNN's assumptions fail. Pre-centred digits. About 30 seconds."""
    import inductive_bias as ib
    df = ib.experiment_no_payoff()
    print("Table C: 8 x 8 digits that somebody already centred and cropped. Translation was solved before the model saw the data, so the assumption has nothing left to buy.")
    return df.round(3) if hasattr(df, "round") else df


# --------------------------------------------------------------------------- Part 3

def a_power(T: int = 60) -> None:
    """Slide: 0.9^60 against 1.1^60."""
    print(f"Printout A: the same number multiplied by itself {T} times")
    for f in (0.5, 0.9, 1.0, 1.1):
        print(f"  {f:.1f} ^ {T} = {f**T:12.6g}")
    print("  Two numbers 0.2 apart give results 170,000 times apart. A message passed through 60 identical steps either dies or explodes.")


def gradient_through_time(T: int = 60) -> pd.DataFrame:
    """Slide: the gradient, measured. How much of the signal at step 1 survives T steps."""
    import sequences as sq
    res = sq.gradient_through_time(T=T)
    rows = res[1] if isinstance(res, tuple) else res
    if hasattr(rows, "columns"):
        df = rows.copy()
    else:
        df = pd.DataFrame({"weight scale": [r["W scale"] for r in rows],
                           "signal left after 10 steps": [f"{r['norms'][9]:.2e}" for r in rows],
                           "signal left after 60 steps": [f"{r['norms'][-1]:.2e}" for r in rows]})
    print("Table B: a real recurrent layer, a real gradient, followed back through 60 steps. At scale 0.5 what reaches step 1 is 7e-17: in 32-bit arithmetic, zero.")
    return df


def half_life() -> pd.DataFrame:
    """Slide: memory half-life for a forget gate f."""
    rows = []
    for f in (0.80, 0.95, 0.99, 0.999):
        hl = np.log(0.5) / np.log(f); one = np.log(0.01) / np.log(f)
        rows.append({"forget gate f": f, "half-life (steps)": round(hl, 1), "1% left after (steps)": int(round(one))})
    print("Table C: an LSTM multiplies its memory by f at every step. f is something it CHOOSES, fresh, each step. Learn f near 1 and the past survives.")
    return pd.DataFrame(rows)


def memory_task(lengths=(10, 40, 90), epochs: int = 100) -> pd.DataFrame:
    """Slide: gates fix propagation. RNN against LSTM at three lengths. About 30 to 60 seconds."""
    import sequences as sq
    df, _ = sq.experiment_memory(lengths=lengths, epochs=epochs)
    cols = ["sequence length", "RNN test acc", "RNN grad at step 1", "LSTM test acc", "LSTM grad at step 1"]
    print("Table D: remember the first symbol across T steps, chance 0.250. Read the row for length 90: the RNN's gradient has collapsed and it is near chance; the LSTM's survives and it is perfect.")
    return df[cols] if cols else df


# --------------------------------------------------------------------------- Part 4

def soft_lookup() -> pd.DataFrame:
    """Slide: every row sums to 1. Six tokens, the verb 'are' looks for its subject 'keys'."""
    import attention as at
    E, scores, weights, out = at.worked_example()
    print("Table A: attention weights. Each ROW is one token asking 'who should I listen to?'. Every row sums to 1.")
    print("         Read the row for 'are': most of its weight is on 'keys', four positions away, not on 'left' next door.")
    w = weights.copy(); w["sum"] = w.sum(axis=1).round(3)
    return w


def four_instructions() -> None:
    """Slide: the formula as four instructions, with the numbers for the token 'are'."""
    import attention as at
    E, scores, weights, out = at.worked_example()
    q = "are"
    print("Printout B: the four instructions, for the token 'are'")
    print(f"  1 PROJECT   'are' becomes a query (what it is looking for: a plural subject); every token offers a key (what it is).")
    print(f"  2 SCORE     query . key for each token:      " + "  ".join(f"{t}={scores.loc[q, t]:.2f}" for t in scores.columns))
    print(f"  3 NORMALISE softmax turns scores into shares: " + "  ".join(f"{t}={weights.loc[q, t]:.2f}" for t in weights.columns) + "   (sum 1.00)")
    print(f"  4 MIX       output for 'are' = those shares x the value vectors: " + ", ".join(f"{v:.2f}" for v in out[list(at.TOKENS).index(q)]))
    print("  Distance never appeared in any of the four steps. 'keys' is four positions away and wins anyway.")


def cannot_see_order() -> pd.DataFrame:
    """Slide: attention cannot see order. Shuffle the tokens, un-shuffle the outputs, compare."""
    import attention as at
    df = at.permutation_test()
    print("Table C: without position information, shuffling the tokens changes the output by 8.88e-16, which is floating-point rounding. With it, the output changes.")
    return df


def the_bill() -> pd.DataFrame:
    """Slide: the bill. Operations against sequence length."""
    import attention as at
    df = at.cost_table()
    print("Table D: attention reaches any distance in ONE step, but its cost grows with the SQUARE of the length. At 32,768 tokens it is 65 times an RNN.")
    return df


# --------------------------------------------------------------------------- Part 5

def the_bargain(sizes=(64, 128, 256, 512), epochs: int = 120) -> pd.DataFrame:
    """Slide: the scaling bargain. CNN against MLP as the data grows. About 90 seconds."""
    import inductive_bias as ib
    df = ib.experiment_data_efficiency(sizes=sizes)
    print("Table A: the same two models as Exercise 1, at four data sizes. The structured model wins by 47 to 75 points at every size here. With 128 times the data (Exercise 1's slide), the unstructured one catches up.")
    return df.round(3) if hasattr(df, "round") else df


def seven_questions() -> None:
    """Slide: seven questions for any architecture."""
    qs = [
        "What is connected to what, in one layer?",
        "Which weights are shared, and across what: space, time, or nothing?",
        "What does that combination assume, as a sentence about the world?",
        "Is that sentence true of my data, and how would I know? (the shuffle test costs one line)",
        "Does my architecture carry the assumption all the way to the output? (0.998 against 0.230)",
        "What is the cost, and where is the crossover?",
        "How much data do I have, relative to the structure to be learned?",
    ]
    print("Printout B: seven questions for anything you design")
    for i, q in enumerate(qs, 1):
        print(f"  {i}. {q}")
