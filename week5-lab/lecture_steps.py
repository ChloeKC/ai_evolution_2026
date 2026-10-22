"""lecture_steps.py: one helper per lecture idea, so each lab step is ONE cell.

    Exercise 1 (Part 1)  two_kinds_of_advice(), the_race(), the_race_without_noise()
    Exercise 2 (Part 2)  check_my_residual(), mark_the_homework(), check_my_loss(),
                         train_with_my_loss(), one_thing_changed(),
                         the_engine(), why_not_finite_differences()
    Exercise 3 (Part 3)  beyond_the_data(), energy_nobody_asked_for()
    Exercise 4 (Part 4)  the_weight_nobody_can_derive(), check_my_constraint(),
                         train_with_my_constraint(), too_few_checkpoints(), when_the_law_is_wrong()
    Exercise 5 (Part 5)  spin_it(), unseen_orientations(), a_false_symmetry(), learn_the_rule(),
                         a_whole_family(), ink_in_water()
    Exercise 6 (Part 6)  a_pde_with_no_data(), match_the_method(), seven_questions()

You never edit this file. The printouts say what was measured and which column to read;
they do not say what it means. Working that out is your part, in SUBMISSION.md.

The check_my_...() functions test the code YOU write in the notebook. They tell you
whether it works and, if not, what to look at. They never print the answer.

Nothing here uses PyTorch or any deep-learning library: the whole lab runs on numpy
and a 250-line automatic-differentiation engine (autodiff.py) that you never open.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)


def _df(res):
    return res[0] if isinstance(res, tuple) else res


# --------------------------------------------------------------------------- Part 1

def two_kinds_of_advice() -> None:
    print("Printout A: two kinds of advice, and what each one does to the list of possible answers")
    print("  A pattern (Week 4):  'people in high-visibility jackets are often construction workers'")
    print("                       -> a nudge. Usually right. Eliminates nothing; every answer stays possible.")
    print("  A law (this week):   'this person is standing in Dublin, so they are not in Cork'")
    print("                       -> a constraint. Removes a whole set of answers, permanently, whatever new evidence arrives.")
    print("  Week 4 gave models the first kind. This week gives them the second.")


def the_race(epochs: int = 3000) -> pd.DataFrame:
    import pinn
    df = _df(pinn.experiment_data_efficiency(epochs=epochs))
    print("Table B: a damped spring. Every row is the SAME network with the SAME optimiser for the SAME number of steps.")
    print("         Read the observations column and the test RMSE column (lower is better).")
    return df


def the_race_without_noise(sizes=(16, 64), epochs: int = 3000) -> pd.DataFrame:
    """Change one thing: the data-only rows again, with PERFECT measurements (no noise)."""
    import pinn
    from autodiff import Node
    sys = pinn.DampedOscillator()
    rows = []
    for n in sizes:
        t_obs, u_obs = pinn.observations(sys, n, noise=0.0, seed=0)
        d = pinn.Solver(sys, seed=0, hard_ic=False)
        pinn.fit(d, epochs=epochs, t_obs=Node(t_obs), u_obs=u_obs)
        rows.append({"model": "data-only, noise-free measurements", "observations": n,
                     "parameters": d.n_params, "test RMSE": round(d.rmse(), 5)})
    print("Table B2: the data-only rows of Table B again, with every measurement exact (noise 0).")
    print("          Compare the test RMSE with the physics-informed row of Table B, which saw no measurements at all.")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Part 2

def mark_the_homework(residual=None) -> pd.DataFrame:
    """Put three candidate answers back into the equation. Pass YOUR residual to mark with it."""
    import pinn
    if residual is not None and _not_written(residual, (np.ones(1), np.ones(1), np.ones(1), 1.0, 0.2, 1.0), "my_residual", "Exercise 2, Build 1"):
        return None
    sys = pinn.DampedOscillator()
    m, c, k = sys.m, sys.c, sys.k
    ts = np.array([0.5, 2.0, 4.0, 7.0])
    eps = 1e-4

    def resid(f):
        u = f(ts); up = (f(ts + eps) - f(ts - eps)) / (2 * eps); upp = (f(ts + eps) - 2 * f(ts) + f(ts - eps)) / eps ** 2
        if residual is not None:
            return np.asarray(residual(u, up, upp, m, c, k), dtype=float).reshape(-1)
        return m * upp + c * up + k * u

    right = resid(sys.exact)
    wrong = resid(lambda t: np.cos(t))
    silly = resid(lambda t: np.ones_like(t) * 0.7)
    df = pd.DataFrame({"time": ts, "true solution": right.round(4), "cos(t), no damping": wrong.round(4), "a flat line at 0.7": silly.round(4)})
    who = "YOUR residual" if residual is not None else "the reference residual"
    print(f"Printout A: each candidate answer put back into  m u'' + c u' + k u, marked with {who}.")
    print("            Each column is one candidate; each number is what is left over at that time. No answer sheet was used.")
    return df


def the_engine() -> None:
    import autodiff
    worst = autodiff.check_engine(verbose=False)
    print("Printout B: the engine's derivatives, checked against hand-derived ones to third order.")
    print(f"            worst relative error = {worst:.2e}   (the rounding error of 64-bit arithmetic; the engine is exact)")


def why_not_finite_differences() -> None:
    import mlp
    print("Printout C: the derivative of a network with respect to its INPUT, two ways (the engine, and a finite difference)")
    r1, r2 = mlp.check_network_derivative(verbose=True)
    print(f"            disagreement on the first derivative: {r1:.1e}   on the second derivative: {r2:.1e}")


def one_thing_changed() -> pd.DataFrame:
    rows = [("the network", "feedforward", "the same"), ("forward pass", "inputs to outputs", "the same"),
            ("the loss", "distance to labels", "residual of an equation  <- the only change"),
            ("backward pass", "d loss / d weights", "the same"), ("optimiser", "Adam", "the same"),
            ("the loop", "guess, check, blame, nudge", "the same")]
    print("Table D: what changed between Weeks 3 and 4 and this week")
    return pd.DataFrame(rows, columns=["component", "Weeks 3 and 4", "Week 5"])


# --------------------------------------------------------------------------- Part 3

def beyond_the_data(t_split: float = 4.0, epochs: int = 3000) -> pd.DataFrame:
    """Measurements only up to t_split (the run lasts 10). Change t_split to move the edge of the data."""
    import pinn
    df = _df(pinn.experiment_extrapolation(t_split=t_split, epochs=epochs))
    print(f"Table A: measurements on the first {t_split * 10:.0f}% of the time only (t from 0 to {t_split:g}); scored on the rest.")
    print("         degradation factor = error beyond the data / error where data existed. 1.0 means no worse.")
    return df


def energy_nobody_asked_for(epochs: int = 3000) -> pd.DataFrame:
    import pinn
    df = _df(pinn.experiment_conservation(epochs=epochs))
    print("Table B: remove the friction, so energy should stay exactly constant. Nobody told any model this.")
    print("         Read the last column: how far each model's energy drifts from the truth.")
    return df


# --------------------------------------------------------------------------- Part 4

def the_weight_nobody_can_derive(epochs: int = 3000) -> pd.DataFrame:
    import pinn
    df = _df(pinn.experiment_constraint_style(epochs=epochs))
    print("Table A: the starting condition u(0) = 1 imposed four ways. Read BOTH the u(0) column and the error column.")
    return df


def too_few_checkpoints(n_points=(10, 50, 200), epochs: int = 2500) -> pd.DataFrame:
    import pinn
    sys = pinn.DampedOscillator()
    rows = []
    rng = np.random.default_rng(0)
    for n in n_points:
        from autodiff import Node
        t_col = Node(np.linspace(0, sys.T, n).reshape(-1, 1))
        solver = pinn.Solver(sys, seed=0)
        out = pinn.fit(solver, epochs=epochs, t_col=t_col)
        hist = out[0] if isinstance(out, tuple) else out
        final_loss = float(hist[-1])
        rows.append({"checkpoints (collocation points)": n, "physics loss it reports": f"{final_loss:.2e}", "true error (RMSE)": round(solver.rmse(), 4)})
    df = pd.DataFrame(rows)
    print("Table B: the same network trained with the equation checked at 10, 50 and 200 places (the checkpoints).")
    print("         Middle column: the physics loss the model reports. Last column: its real error against the truth.")
    return df


def when_the_law_is_wrong(c_assumed=(0.2, 0.4, 0.8), epochs: int = 3000) -> pd.DataFrame:
    """The physics rows are told these friction values. The truth is 0.2. Add your own value to the tuple."""
    import pinn
    df = _df(pinn.experiment_wrong_physics(c_assumed=tuple(c_assumed), epochs=epochs))
    print("Table C: the true friction is c = 0.2. Every model gets the same real measurements. The physics rows are also TOLD a value of c.")
    print("         Read the true RMSE column, then the physics-loss column (what the model itself would show you).")
    return df


# --------------------------------------------------------------------------- Part 5

def spin_it() -> pd.DataFrame:
    import symmetry
    df = _df(symmetry.experiment_invariance()).drop(columns=["verdict"], errors="ignore")
    print("Table A: three particles whose energy depends only on the distances between them. Rotate the whole arrangement")
    print("         and compare each model's answer before and after.")
    return df


def unseen_orientations() -> pd.DataFrame:
    import symmetry
    df = _df(symmetry.experiment_unseen_orientations(quick=True))
    print("Table B: trained on some orientations, tested on all of them. Read the degradation column.")
    return df


def a_false_symmetry() -> pd.DataFrame:
    import symmetry
    df = _df(symmetry.experiment_wrong_symmetry(quick=True)).drop(columns=["verdict"], errors="ignore")
    print("Table C: now add gravity, so turning the arrangement upside down really does change the energy.")
    print("         Compare the two models' errors.")
    return df


def learn_the_rule() -> pd.DataFrame:
    import neural_ode
    df = _df(neural_ode.experiment_rollout(quick=True))
    print("Table D: trained on a window of time, then run three times further than that window. Read the degradation column.")
    return df


def a_whole_family() -> pd.DataFrame:
    import operators
    df = _df(operators.experiment_operator(quick=True))
    print("Table E: a new friction value, solved two ways. Read the time per new problem and the error.")
    return df


def ink_in_water() -> pd.DataFrame:
    import diffusion
    df = _df(diffusion.experiment_equilibrium())
    print("Table F: add a little noise, again and again (nothing is trained here). Last column: how far the data is from pure noise.")
    return df


# --------------------------------------------------------------------------- Part 6

def a_pde_with_no_data() -> pd.DataFrame:
    import heat
    df = _df(heat.experiment_heat(quick=True))
    print("Table A: heat spreading along a rod (two inputs: position and time). The physics-informed row again has 0 observations.")
    return df


_MATCH = [
    ("an equation, one situation to solve", "a physics-informed network (PINN)", "needs no data; gives the answer and its slopes"),
    ("an equation, thousands of situations", "a neural operator", "train once, then each new case costs a fraction of a millisecond"),
    ("an equation on a standard, simple domain", "a classical numerical solver", "faster, more accurate, with error guarantees"),
    ("an equation with a number you do not know", "a PINN, run backwards", "make the unknown number a trainable dial"),
    ("half the rule known, half unknown", "a structured neural ODE", "impose the known half, learn the rest"),
    ("an exact symmetry", "an invariant or equivariant model", "an exact guarantee and far less data needed"),
    ("no reliable physics at all", "ordinary machine learning", "a guessed law is worse than none"),
]
_SHUFFLE = [3, 6, 0, 5, 2, 4, 1]          # fixed, so every student sees the same letters


def match_the_method(reveal: bool = False) -> pd.DataFrame:
    """reveal=False: the situations and the methods, jumbled. Match them in SUBMISSION.md first.
    reveal=True: the reference matching, with the letters, so you can mark your own."""
    letters = "ABCDEFG"
    method_letter = {_MATCH[j][1]: letters[i] for i, j in enumerate(_SHUFFLE)}
    if not reveal:
        print("Table B: match each situation (1 to 7) to a method (A to G). Each method is used once.")
        print("         Write your matches in SUBMISSION.md, then run  ls.match_the_method(reveal=True)  to mark them.")
        left = pd.DataFrame({"situation": [r[0] for r in _MATCH]}, index=range(1, 8))
        right = pd.DataFrame({"method": [_MATCH[j][1] for j in _SHUFFLE]}, index=list(letters))
        try:
            from IPython.display import display
            display(left); display(right)
        except Exception:
            print(left.to_string()); print(right.to_string())
        return None
    print("Table B (revealed): what you know decides the method. Mark your matches; for any you missed, read the 'because' column.")
    return pd.DataFrame([(r[0], method_letter[r[1]] + "  " + r[1], r[2]) for r in _MATCH],
                        columns=["if you have", "reach for", "because"], index=range(1, 8))


def seven_questions() -> None:
    qs = ["What do I know, as a sentence a non-specialist would understand?",
          "Is it exactly true, approximately true, or a guess?",
          "Is it a hard constraint (built into the arithmetic) or a soft penalty (a term in the loss)?",
          "Where is the loss actually checked, and what does that leave unchecked?",
          "What consequence can I check that I did not train on? (energy is the classic)",
          "What happens if my law is wrong by 20%?",
          "Would a classical solver, or ordinary machine learning, have done this better?"]
    print("Printout C: seven questions for anything you build with a law in it")
    for i, q in enumerate(qs, 1):
        print(f"  {i}. {q}")


# =========================================================================== YOUR CODE
# The checks below test the functions you write in the notebook. They say whether
# your code works, and if not, what to look at. They never print the answer.

def _ok(msg):
    print("  \u2713 " + msg)


def _no(msg):
    print("  \u2717 " + msg)


def _num(x):
    """A plain float from whatever the student's function returned (number, array or Node)."""
    v = getattr(x, "value", x)
    return float(np.asarray(v, dtype=float).reshape(-1)[0])


def _not_written(fn, args, name, build):
    """True (after saying so) if a function from the notebook still returns None."""
    try:
        out = fn(*args)
    except Exception:
        return False                      # let the real call report the error
    if out is None:
        print(f"{name} still returns None: write it first ({build}), and run its check cell until every line shows \u2713.")
        return True
    return False


def check_my_residual(my_residual) -> None:
    """Exercise 2, Build 1. Is my_residual(u, u_t, u_tt, m, c, k) the leftover of m u'' + c u' + k u = 0?"""
    from autodiff import Node
    import pinn
    print("Check: my_residual")
    m, c, k = 2.0, 0.3, 5.0          # deliberately not the spring's own 1, 0.2, 1: a swapped coefficient shows up
    one, zero = np.array([1.0]), np.array([0.0])
    try:
        r = my_residual(one, zero, zero, m, c, k)
    except Exception as e:
        _no(f"calling it raised {type(e).__name__}: {e}")
        print("    Its inputs are u, u_t, u_tt (the guess, its slope, the slope of its slope) and the numbers m, c, k.")
        return
    if r is None:
        _no("it returns None. Replace `return None` with your one line.")
        return
    probes = [((1, 0, 0), "only u is 1 (u' and u'' are 0)", k, "u"),
              ((0, 1, 0), "only u' is 1", c, "u'"),
              ((0, 0, 1), "only u'' is 1", m, "u''")]
    all_ok = True
    for (a, b, d), words, want, term in probes:
        got = _num(my_residual(np.array([a * 1.0]), np.array([b * 1.0]), np.array([d * 1.0]), m, c, k))
        if abs(got - want) > 1e-9:
            all_ok = False
            _no(f"when {words}, with m = 2, c = 0.3, k = 5, your leftover is {got:g}. "
                f"In  m u'' + c u' + k u, which number multiplies {term}?")
    if not all_ok:
        return
    _ok("each of the three terms has the right coefficient")
    got = _num(my_residual(one, one, one, m, c, k))
    if abs(got - (m + c + k)) > 1e-9:
        _no(f"with u, u', u'' all equal to 1 your leftover is {got:g}: the three terms are not simply added together")
        return
    _ok("the three terms are added together")
    # the true solution must leave (almost) nothing, a wrong one must leave something
    sys = pinn.DampedOscillator()
    t = np.linspace(0.1, 9.9, 50)
    eps = 1e-4
    f = sys.exact
    u, ut, utt = f(t), (f(t + eps) - f(t - eps)) / (2 * eps), (f(t + eps) - 2 * f(t) + f(t - eps)) / eps ** 2
    r_true = np.asarray(getattr(my_residual(u, ut, utt, sys.m, sys.c, sys.k), "value",
                                my_residual(u, ut, utt, sys.m, sys.c, sys.k)), float)
    if np.max(np.abs(r_true)) > 1e-5:
        _no(f"for the true solution of the spring your leftover reaches {np.max(np.abs(r_true)):.2g}; it should be 0")
        return
    _ok(f"for the true solution of the spring the leftover is {np.max(np.abs(r_true)):.0e} everywhere (zero, up to rounding)")
    # it must also work on the engine's values, which is what training uses
    try:
        rn = my_residual(Node(u.reshape(-1, 1)), Node(ut.reshape(-1, 1)), Node(utt.reshape(-1, 1)), sys.m, sys.c, sys.k)
        assert isinstance(rn, Node), "it did not return one of the engine's values"
    except Exception as e:
        _no(f"it works on plain numbers but not on the engine's values ({e}). Use only + - * on u, u_t, u_tt; "
            "numpy functions such as np.multiply step outside the engine, so training could not take slopes through them.")
        return
    _ok("it also works on the engine's values, so it can be trained through")
    print("  All checks passed. Next: mark the homework with it.")


def check_my_loss(my_physics_loss) -> None:
    """Exercise 2, Build 2. Is my_physics_loss(residual) one number that is zero only when the residual is zero everywhere?"""
    from autodiff import Node, grad
    print("Check: my_physics_loss")
    def L(vals):
        return my_physics_loss(Node(np.array(vals, dtype=float).reshape(-1, 1)))
    try:
        out = L([1.0, -1.0])
    except Exception as e:
        _no(f"calling it raised {type(e).__name__}: {e}")
        print("    Its input is the residual at many times at once. You have mean(...) available; ** and * work too.")
        return
    if out is None:
        _no("it returns None. Replace `return None` with your one line.")
        return
    if not isinstance(out, Node):
        _no("it returns a plain number, not one of the engine's values, so training cannot take slopes through it. "
            "Use mean(...) from the first cell, not np.mean.")
        return
    if np.asarray(out.value).size != 1:
        _no(f"it returns {np.asarray(out.value).size} numbers. A loss is ONE number for the whole set of times.")
        return
    _ok("it returns one number, inside the engine")
    z = _num(L([0.0, 0.0, 0.0]))
    if abs(z) > 1e-12:
        _no(f"when the leftover is zero at every time, your loss is {z:g}. It should be 0: nothing is wrong.")
        return
    _ok("zero leftover gives zero loss")
    v = _num(out)
    if v <= 1e-12:
        _no(f"a leftover of +1 at one time and -1 at another gives your loss {v:g}. The two errors cancelled, so the "
            "network would be told it was perfect. What can you do to each leftover before averaging so that cannot happen?")
        return
    if _num(L([2.0, 2.0])) <= _num(L([1.0, 1.0])):
        _no("a leftover of 2 everywhere does not score worse than a leftover of 1 everywhere")
        return
    if _num(L([-3.0])) < 0:
        _no("a negative leftover gives a negative loss")
        return
    _ok("errors of opposite sign do not cancel, and bigger leftovers score worse")
    r = Node(np.array([[0.5], [-2.0]]))
    try:
        g = grad(my_physics_loss(r), r).value
        assert np.all(np.isfinite(g))
    except Exception as e:
        _no(f"the engine could not take its slope ({e})")
        return
    _ok("the engine can take its slope, so it can be trained")
    print("  All checks passed. Next: train a network with it.")


def _reference_loop_text(loss_line):
    return ("    for step in range(3000):                       # the Week 3 loop, unchanged\n"
            "        u    = network(t)                          # guess\n"
            "        u_t  = slope of u;  u_tt = slope of u_t     # the engine, exactly\n"
            f"        loss = {loss_line}   # check  <- YOUR code\n"
            "        nudge every weight downhill (Adam)          # blame and nudge")


def _plot(solver, sys, title):
    import matplotlib.pyplot as plt
    t = np.linspace(0, sys.T, 400).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.plot(t, sys.exact(t), color="#47546B", lw=3, alpha=.45, label="the true swing")
    ax.plot(t, solver.predict(t), color="#A96E08", lw=1.6, ls="--", label="your network")
    ax.set_xlabel("time"); ax.set_ylabel("position"); ax.set_title(title, fontsize=10, color="#47546B")
    ax.legend(frameon=False, fontsize=9); fig.tight_layout(); plt.show()


def _train(solver, sys, my_residual, my_physics_loss, epochs):
    from autodiff import Node, grad, sum_all
    from mlp import Adam
    t_col = Node(np.linspace(0, sys.T, 200).reshape(-1, 1))
    opt = Adam(solver.params, lr=3e-3)
    with np.errstate(all="ignore"):
        for ep in range(epochs):
            u = solver(t_col)
            u_t = grad(sum_all(u), t_col)
            u_tt = grad(sum_all(u_t), t_col)
            loss = my_physics_loss(my_residual(u, u_t, u_tt, sys.m, sys.c, sys.k))
            opt.step(grad(loss, solver.params))
            if ep + 1 in (int(epochs * 0.5), int(epochs * 0.8)):
                opt.set_lr(opt.lr * 0.3)
        err = solver.rmse()
    return float(loss.value.item()), err


def train_with_my_loss(my_residual, my_physics_loss, epochs: int = 3000) -> pd.DataFrame:
    """Exercise 2. A physics-informed network trained with YOUR residual and YOUR loss, and no measurements."""
    import pinn
    from autodiff import Node
    if _not_written(my_residual, (np.ones(1), np.ones(1), np.ones(1), 1.0, 0.2, 1.0), "my_residual", "Exercise 2, Build 1") or \
       _not_written(my_physics_loss, (Node(np.ones((2, 1))),), "my_physics_loss", "Exercise 2, Build 2"):
        return None
    sys = pinn.DampedOscillator()
    s = pinn.Solver(sys, seed=0)            # the starting condition is built in for you here; Exercise 4 has you build it
    print("Training a network with no measurements, using your two functions. About 20 seconds.")
    print(_reference_loop_text("my_physics_loss(my_residual(u, u_t, u_tt, m, c, k))"))
    loss, err = _train(s, sys, my_residual, my_physics_loss, epochs)
    if not np.isfinite(err):
        print("  The training produced NaN (not a number). Run check_my_residual and check_my_loss again.")
        return None
    _plot(s, sys, "Your physics-informed network, trained on zero measurements")
    print("Table G: your network. Compare its test RMSE with the physics-informed row of Exercise 1, Table B.")
    return pd.DataFrame([{"model": "YOUR physics loss (no data)", "observations": 0, "parameters": s.n_params,
                          "final loss": f"{loss:.2e}", "test RMSE": round(err, 5)}])


class _MySolver:
    """The lab's network, with YOUR way of building the starting condition into the guess."""
    def __init__(self, system, my_guess, seed=0):
        import pinn
        self._inner = pinn.Solver(system, seed=seed, hard_ic=False)
        self.sys, self.net, self.params, self.guess = system, self._inner.net, self._inner.params, my_guess
        self.n_params = self._inner.n_params

    def __call__(self, t):
        from autodiff import const, add, mul
        t = const(t)
        raw = self.net(add(mul(t, 2.0 / self.sys.T), -1.0))
        return self.guess(t, raw)

    def rmse(self, n=600):
        from autodiff import Node
        tg = np.linspace(0, self.sys.T, n).reshape(-1, 1)
        return float(np.sqrt(np.mean((self(Node(tg)).value - self.sys.exact(tg)) ** 2)))

    def predict(self, t):
        from autodiff import Node
        return self(Node(np.asarray(t, float).reshape(-1, 1))).value


def _factor_at(my_guess, t):
    """How strongly the network's output is multiplied at time t (assumes the guess is linear in N)."""
    from autodiff import Node
    T = Node(np.array([[t]]))
    return _num(my_guess(T, Node(np.array([[1.0]])))) - _num(my_guess(T, Node(np.array([[0.0]]))))


def check_my_constraint(my_guess) -> None:
    """Exercise 4, Build 3. Does my_guess(t, N) give u(0) = 1 and u'(0) = 0 for EVERY network output N?"""
    from autodiff import Node, grad, sum_all
    print("Check: my_guess")
    rng = np.random.default_rng(0)
    starts, speeds, third = [], [], []
    with np.errstate(all="ignore"):
        for _ in range(4):
            a, b, c = rng.normal(0, 1, 3)
            t0 = Node(np.array([[0.0]]))
            N = a + b * t0 + c * t0 * t0          # stands in for a network: any value, any slope at the start
            try:
                u = my_guess(t0, N)
            except Exception as e:
                _no(f"calling it raised {type(e).__name__}: {e}")
                print("    t is the time and N is the network's output at that time. Use only + - * / and numbers.")
                return
            if u is None:
                _no("it returns None. Replace `return None` with your guess.")
                return
            if not isinstance(u, Node):
                _no("it returns a plain number, not one of the engine's values: use t and N with + - * /, not numpy functions.")
                return
            starts.append(_num(u))
            u_t = grad(sum_all(u), t0)
            speeds.append(_num(u_t))
            u_tt = grad(sum_all(u_t), t0)
            third.append(_num(grad(sum_all(u_tt), t0)))
        t5 = Node(np.array([[5.0]]))
        moves = abs(_num(my_guess(t5, Node(np.array([[1.0]])))) - _num(my_guess(t5, Node(np.array([[0.0]])))))
    if moves < 1e-9:
        _no("your guess is the same whatever the network outputs, so the network can never learn anything. "
            "The network's output N has to appear in it.")
        return
    if not all(np.isfinite(starts + speeds + third)):
        _no("a slope at t = 0 comes out as not-a-number. If you wrote t**2, write t*t instead: the engine's power rule "
            "divides by t when training takes slopes of slopes, and t is zero at the start (see square() in autodiff.py).")
        return
    if max(abs(x - 1.0) for x in starts) > 1e-9:
        _no("for four different network outputs your guess starts at " + ", ".join(f"{x:.3f}" for x in starts) +
            ". The start must be 1 whatever the network does: what could you multiply N by so it has no say at t = 0?")
        return
    _ok("u(0) = 1 for every network output tried")
    if max(abs(x) for x in speeds) > 1e-9:
        _no("but the starting speed u'(0) is " + ", ".join(f"{x:.3f}" for x in speeds) +
            " for the four network outputs; it must be 0. Something that is zero at t = 0 is not enough: "
            "its slope at t = 0 has to be zero as well.")
        return
    _ok("u'(0) = 0 for every network output tried")
    with np.errstate(all="ignore"):
        f10 = _factor_at(my_guess, 10.0)
    print(f"  Both conditions are built in, for any weights, before any training. "
          f"(For later: at t = 10 the network's output is multiplied by {f10:.3g}.)")
    print("  All checks passed. Next: train with it.")


def train_with_my_constraint(my_guess, my_residual, my_physics_loss, epochs: int = 3000) -> pd.DataFrame:
    """Exercise 4. A network trained with YOUR guess (Build 3), YOUR residual and YOUR loss (Exercise 2)."""
    import pinn
    from autodiff import Node
    if _not_written(my_guess, (Node(np.ones((1, 1))), Node(np.ones((1, 1)))), "my_guess", "Exercise 4, Build 3") or \
       _not_written(my_residual, (np.ones(1), np.ones(1), np.ones(1), 1.0, 0.2, 1.0), "my_residual", "Exercise 2, Build 1") or \
       _not_written(my_physics_loss, (Node(np.ones((2, 1))),), "my_physics_loss", "Exercise 2, Build 2"):
        return None
    sys = pinn.DampedOscillator()
    s = _MySolver(sys, my_guess, seed=0)
    print("Training with your starting condition, your residual and your loss: everything except the network and the")
    print("loop is now your code. No measurements. About 20 seconds.")
    loss, err = _train(s, sys, my_residual, my_physics_loss, epochs)
    if not np.isfinite(err):
        print("  The training produced NaN (not a number). Run check_my_constraint again; if you wrote t**2, write t*t.")
        return None
    _plot(s, sys, "Your network, your starting condition, zero measurements")
    with np.errstate(all="ignore"):
        f10 = _factor_at(my_guess, sys.T)
    print("Table A2: compare with the hard-constraint row of Table A.")
    if err > 0.005:
        print(f"  Both conditions hold, yet the error is about {err / 0.00135:.0f} times the reference hard constraint's.")
        print(f"  At t = 10 your guess multiplies the network's output by {f10:.3g}. What does that ask of the network out")
        print("  there, and of the slopes the residual needs? Is there a factor with the same two properties at t = 0")
        print("  that stays near 1 later on? (Optional: try it. Each attempt is about 20 seconds.)")
    return pd.DataFrame([{"model": "YOUR hard constraint (no data)", "factor on N at t = 10": round(f10, 3),
                          "final loss": f"{loss:.2e}", "test RMSE": round(err, 5)}])
