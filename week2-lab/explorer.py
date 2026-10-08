"""
explorer.py, the Bias & Fairness Explorer.

The Evolution of Machine Learning and AI · Week 2 Lab
South East Technological University

Four figures, each of which corresponds to a slide you saw in the lecture:

    fig_distributions   two groups, one model, one line   (Chapter 7)
    fig_threshold_sweep every criterion, every threshold  (Chapters 7-9)
    fig_calibration     what the score means, per group   (Chapters 8-9)
    fig_impossibility   satisfy one criterion, break two  (Chapter 9)

There is also an optional ipywidgets slider (`interactive_explorer`) for use in
Colab or Jupyter, which is the version used in the two-hour lab.

Run directly to regenerate all four figures as PNGs:

    python explorer.py
"""

from __future__ import annotations

from pathlib import Path

import sys
import matplotlib
if "ipykernel" not in sys.modules:      # keep inline plots working in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fairness as fm

# SETU palette, matching the lecture slides and the booklet.
SLATE = "#176B78"
AMBER = "#C98200"
RED = "#B63D32"
BLUE = "#1677A6"
GREY = "#6F858A"
GREEN = "#1E6F52"
GROUP_COLOURS = {"A": SLATE, "B": AMBER}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelcolor": "#12343B",
    "text.color": "#12343B",
    "axes.edgecolor": GREY,
    "xtick.color": GREY,
    "ytick.color": GREY,
})


# ----------------------------------------------------------------------------

def fig_distributions(y, s, a, t=0.5, ax=None):
    """Score distributions by group, with the threshold and selection rates.

    This is the single most important picture in the week. One identical rule,
    applied to two populations whose recorded circumstances differ, produces
    two very different selection rates. Nothing in the code is unequal.
    """
    a = np.asarray(a)
    s = np.asarray(s, dtype=float)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7.2, 3.4))

    bins = np.linspace(0, 1, 46)
    for g in sorted(pd.unique(a)):
        m = a == g
        ax.hist(s[m], bins=bins, density=True, alpha=0.55,
                color=GROUP_COLOURS.get(g, GREY), label=f"group {g}  (n={m.sum()})")

    ax.axvline(t, color=RED, ls="--", lw=2)
    ax.text(t, ax.get_ylim()[1] * 0.98, f"  t = {t:.2f}", color=RED,
            va="top", ha="left", fontweight="bold", fontsize=9)

    parts = []
    for g in sorted(pd.unique(a)):
        m = a == g
        parts.append(f"group {g}: {(s[m] >= t).mean():.0%} selected")
    ax.set_title("Same model, same rule, different outcomes\n" + "   |   ".join(parts))
    ax.set_xlabel("model score (probability of repayment)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    if created:
        plt.tight_layout()
    return ax.figure


def fig_threshold_sweep(y, s, a, ax=None):
    """Every fairness gap, at every threshold, on one pair of axes.

    Read it as the lab's central finding: the curves do not reach zero at the
    same place. Wherever you put the threshold, you are choosing which
    criterion to satisfy and which to sacrifice.
    """
    sweep = fm.threshold_sweep(y, s, a)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7.2, 4.0))

    series = [
        ("demographic_parity_diff", "demographic parity gap", SLATE, "-"),
        ("equal_opportunity_diff", "equal opportunity gap (TPR)", AMBER, "-"),
        ("equalised_odds_diff", "equalised odds gap", RED, "--"),
        ("predictive_parity_diff", "predictive parity gap (PPV)", BLUE, "-."),
    ]
    for col, label, colour, ls in series:
        ax.plot(sweep.threshold, sweep[col], color=colour, ls=ls, lw=1.9, label=label)
        best = sweep.loc[sweep[col].idxmin()]
        ax.plot(best.threshold, best[col], "o", color=colour, ms=5)

    ax2 = ax.twinx()
    ax2.plot(sweep.threshold, sweep.accuracy, color=GREY, lw=1.2, alpha=0.8)
    ax2.set_ylabel("overall accuracy", color=GREY)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, 1)

    ax.set_xlabel("decision threshold")
    ax.set_ylabel("gap between groups (0 = parity)")
    ax.set_ylim(0, None)
    ax.set_title("No single threshold satisfies every criterion\n"
                 "(dots mark each criterion's best threshold; grey line is accuracy)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    if created:
        plt.tight_layout()
    return ax.figure, sweep


def fig_calibration(y, s, a, n_bins=10, ax=None):
    """Observed outcome rate against predicted score, per group.

    On the diagonal means the score means what it says. This is the criterion
    the COMPAS vendor defended, and Chapter 9 explains why satisfying it forces
    the error rates apart when base rates differ.
    """
    cc = fm.calibration_curve_by_group(y, s, a, n_bins=n_bins)
    ece = fm.calibration_error_by_group(y, s, a, n_bins=n_bins).set_index("group").ece

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(4.6, 4.2))

    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1, label="perfect calibration")
    for g, sub in cc.groupby("group"):
        ax.plot(sub.mean_score, sub.observed_rate, "o-", lw=1.8, ms=4,
                color=GROUP_COLOURS.get(g, GREY),
                label=f"group {g}  (ECE {ece[g]:.3f})")
    ax.set_xlabel("predicted score")
    ax.set_ylabel("observed repayment rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Does a score of 0.7 mean 0.7 for everybody?")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    if created:
        plt.tight_layout()
    return ax.figure


def fig_impossibility(y, s, a):
    """Satisfy one criterion at a time; show what the other two do.

    This is the lab's proof-by-experiment of Chapter 9. For each criterion we
    search for the per-group thresholds that best satisfy *it*, then measure
    the other criteria under those same thresholds. No configuration zeroes
    all of them, because none exists.
    """
    a = np.asarray(a)
    groups = sorted(pd.unique(a))
    grid = np.linspace(0.05, 0.95, 61)

    def gaps(thr: dict) -> dict:
        df = fm.metrics_by_group(y, s, a, thr)
        per = df[df.group != "ALL"]
        return {
            "selection rate\n(demographic parity)": float(per.selection_rate.max() - per.selection_rate.min()),
            "false-positive rate\n(equalised odds)": float(per.fpr.max() - per.fpr.min()),
            "predictive value\n(calibration/PPV)": float(per.ppv.max() - per.ppv.min()),
            "_accuracy": float(df[df.group == "ALL"].accuracy.iloc[0]),
        }

    targets = list(gaps({g: 0.5 for g in groups}).keys())[:3]
    results = {}
    for target in targets:
        best, best_val = None, np.inf
        for t0 in grid:
            for t1 in grid:
                thr = {groups[0]: float(t0), groups[1]: float(t1)}
                v = gaps(thr)
                # tie-break on accuracy so we do not pick a degenerate threshold
                score = v[target] - 0.02 * v["_accuracy"]
                if score < best_val:
                    best_val, best, best_full = score, thr, v
        results[target] = (best, best_full)

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    width = 0.24
    xs = np.arange(len(targets))
    colours = [SLATE, AMBER, BLUE]

    for i, target in enumerate(targets):
        thr, vals = results[target]
        for j, measured in enumerate(targets):
            v = vals[measured]
            bar = ax.bar(xs[j] + (i - 1) * width, v, width,
                         color=colours[i], alpha=1.0 if i == j else 0.42,
                         edgecolor="white")
            if i == j:
                ax.text(xs[j] + (i - 1) * width, v + 0.008, "optimised",
                        ha="center", fontsize=7, color=colours[i], fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(targets, fontsize=8)
    ax.set_ylabel("gap between groups")
    ax.set_title("Optimise any one criterion (solid bar) and the others do not follow\n"
                 "Three colours = three optimisation targets, each with its own thresholds")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[i]) for i in range(3)]
    ax.legend(handles, [f"thresholds tuned for {t.splitlines()[0]}" for t in targets],
              frameon=False, fontsize=7.5, loc="upper right")
    plt.tight_layout()

    table = pd.DataFrame([
        {"optimised_for": t.replace("\n", " "),
         **{k.replace("\n", " "): round(v, 4) for k, v in vals.items() if k != "_accuracy"},
         "accuracy": round(vals["_accuracy"], 4),
         "thresholds": {g: round(x, 3) for g, x in thr.items()}}
        for t, (thr, vals) in results.items()
    ])
    return fig, table


# ----------------------------------------------------------------------------
# Notebook slider (optional)
# ----------------------------------------------------------------------------

def interactive_explorer(y, s, a):
    """ipywidgets threshold slider. Falls back to a static figure if missing."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets not installed; showing the static version instead.")
        print("  pip install ipywidgets")
        return fig_distributions(y, s, a, 0.5)

    slider = widgets.FloatSlider(value=0.5, min=0.05, max=0.95, step=0.01,
                                 description="threshold", continuous_update=False,
                                 layout=widgets.Layout(width="70%"))
    out = widgets.Output()

    def redraw(change=None):
        with out:
            out.clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
            fig_distributions(y, s, a, slider.value, ax=axes[0])
            df = fm.metrics_by_group(y, s, a, slider.value)
            per = df[df.group != "ALL"]
            names = ["selection rate", "TPR", "FPR", "PPV"]
            gaps = [
                per.selection_rate.max() - per.selection_rate.min(),
                per.tpr.max() - per.tpr.min(),
                per.fpr.max() - per.fpr.min(),
                per.ppv.max() - per.ppv.min(),
            ]
            axes[1].barh(names, gaps, color=[SLATE, AMBER, RED, BLUE])
            axes[1].set_xlim(0, 0.6)
            axes[1].set_xlabel("gap between groups")
            axes[1].set_title("Move the slider. Watch them fight.")
            plt.tight_layout()
            plt.show()

    slider.observe(redraw, names="value")
    redraw()
    display(widgets.VBox([slider, out]))


# ----------------------------------------------------------------------------

def main(outdir: str = "figures"):
    from datasets import make_lending_data, train_model

    out = Path(outdir)
    out.mkdir(exist_ok=True)

    df = make_lending_data(n=8000, seed=1)
    _, _, test, scores = train_model(df, seed=1)
    y, a = test.repaid.to_numpy(), test.group.to_numpy()

    fig_distributions(y, scores, a, 0.5).savefig(out / "01_distributions.png", bbox_inches="tight")
    f, sweep = fig_threshold_sweep(y, scores, a)
    f.savefig(out / "02_threshold_sweep.png", bbox_inches="tight")
    fig_calibration(y, scores, a).savefig(out / "03_calibration.png", bbox_inches="tight")
    f, table = fig_impossibility(y, scores, a)
    f.savefig(out / "04_impossibility.png", bbox_inches="tight")

    print(fm.summarise(y, scores, a, 0.5))
    # The degenerate solutions are worth showing on purpose: you can always
    # achieve perfect parity by approving nobody, or everybody. Any fairness
    # result that does not report the selection rate alongside the gap should
    # be treated as unreported until it does.
    print("\nBest threshold for each criterion, UNCONSTRAINED:")
    for col in ["demographic_parity_diff", "equal_opportunity_diff",
                "equalised_odds_diff", "predictive_parity_diff"]:
        r = sweep.loc[sweep[col].idxmin()]
        flag = "  <-- degenerate" if not (0.10 <= r.selection_rate <= 0.90) else ""
        print(f"  {col:26s} t = {r.threshold:.2f}   gap = {r[col]:.3f}   "
              f"acc = {r.accuracy:.3f}   selects {r.selection_rate:.0%}{flag}")

    usable = sweep[(sweep.selection_rate >= 0.10) & (sweep.selection_rate <= 0.90)]
    print("\nBest threshold within a usable operating range (10-90% approved):")
    for col in ["demographic_parity_diff", "equal_opportunity_diff",
                "equalised_odds_diff", "predictive_parity_diff"]:
        r = usable.loc[usable[col].idxmin()]
        print(f"  {col:26s} t = {r.threshold:.2f}   gap = {r[col]:.3f}   "
              f"acc = {r.accuracy:.3f}   selects {r.selection_rate:.0%}")
    print("\nImpossibility demonstration:")
    print(table.to_string(index=False))
    print(f"\nFigures written to {out.resolve()}")


if __name__ == "__main__":
    main()
