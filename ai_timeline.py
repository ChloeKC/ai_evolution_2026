"""
ai_timeline.py — the AI Evolution Timeline, as figures.

The Evolution of Machine Learning and AI · Week 1 Lab
South East Technological University

The 30-minute macro orientation from the lab brief, as four figures you can
regenerate, argue with, and edit:

    fig_hype_wave        the one picture to remember      (booklet Ch. 1)
    fig_family_fortunes  symbolic vs connectionist        (booklet Ch. 4)
    fig_ingredients      when each supply arrived         (booklet Ch. 10)
    fig_era_bands        the master timeline as a chart   (booklet Ch. 15)

    python ai_timeline.py        # writes all four to ./figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from timeline_data import ERAS, FIRST_COMBINATION, SUPPLY, WINTERS, ingredients_available, load

SLATE, AMBER, RED, BLUE, GREY, GREEN = "#47546B", "#A96E08", "#9E3B2C", "#4E7291", "#8A93A5", "#2E7D4F"
FAMILY_COLOUR = {"symbolic": SLATE, "connectionist": AMBER,
                 "statistical": BLUE, "neither": GREY}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
    "text.color": "#22293A", "axes.labelcolor": "#22293A",
})

# The enthusiasm-and-funding curve from the lecture, as control points.
WAVE = [(1950, .10), (1958, .52), (1966, .63), (1969, .58), (1974, .10),
        (1978, .10), (1984, .58), (1986, .62), (1988, .50), (1991, .12),
        (1995, .13), (2000, .20), (2005, .27), (2012, .55), (2017, .72),
        (2020, .82), (2022, .96), (2024, .90), (2026, .88)]


def _wave(years):
    xs = np.array([p[0] for p in WAVE], dtype=float)
    ys = np.array([p[1] for p in WAVE], dtype=float)
    return np.interp(years, xs, ys)


def fig_hype_wave(annotate: int = 8, upto: int = 2026, ax=None):
    """Enthusiasm and funding, 1950 to now, with the two winters shaded."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(10, 3.8))

    yrs = np.linspace(1950, upto, 600)
    ax.plot(yrs, _wave(yrs), color=SLATE, lw=2.6)
    ax.fill_between(yrs, 0, _wave(yrs), color=SLATE, alpha=0.07)

    for a, b, label in WINTERS:
        if upto > a:
            ax.axvspan(a, min(b, upto), color=BLUE, alpha=0.16)
            ax.text((a + min(b, upto)) / 2, 0.94, label, ha="center", fontsize=8,
                    color=BLUE, fontweight="bold")

    df = load()
    big = df[df.year >= 1950].copy()
    big["impact"] = big.kind.map({"shock": 3, "system": 2, "idea": 2,
                                  "dataset": 2, "hardware": 2, "institution": 1})
    picks = (big[big.year <= upto]
             .sort_values(["impact", "year"], ascending=[False, True])
             .drop_duplicates("year").head(annotate).sort_values("year"))

    for i, (_, r) in enumerate(picks.iterrows()):
        y = float(_wave(np.array([r.year]))[0])
        ax.plot(r.year, y, "o", color=FAMILY_COLOUR[r.family], ms=6, zorder=5,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(f"{r['name']}\n{r.year}", (r.year, y),
                    xytext=(0, 26 if i % 2 == 0 else -34), textcoords="offset points",
                    ha="center", fontsize=7.2, color="#22293A",
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))

    ax.set_ylim(0, 1.08); ax.set_xlim(1950, upto)
    ax.set_yticks([])
    ax.set_ylabel("enthusiasm and funding")
    ax.set_title("The one picture to remember: the field moves in cycles")
    if created:
        plt.tight_layout()
    return ax.figure


def fig_family_fortunes(ax=None):
    """Milestones per decade, by family. The pendulum, counted.

    Symbolic AI dominates into the 1990s and crashes twice; connectionism
    spends thirty years as the losing side and then wins nearly everything.
    """
    df = load()
    df = df[df.year >= 1940].copy()
    df["decade"] = (df.year // 10) * 10
    counts = (df.pivot_table(index="decade", columns="family", values="name",
                             aggfunc="count").fillna(0))
    for fam in ["symbolic", "connectionist", "statistical", "neither"]:
        if fam not in counts:
            counts[fam] = 0

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8.2, 3.6))

    x = np.arange(len(counts.index)); w = 0.26
    ax.bar(x - w, counts["symbolic"], w, label="symbolic (rules by hand)", color=SLATE)
    ax.bar(x, counts["connectionist"], w, label="connectionist (learned from data)", color=AMBER)
    ax.bar(x + w, counts["statistical"], w, label="statistical", color=BLUE)

    for a, b, _ in WINTERS:
        ax.axvspan((a - 1940) / 10 - 0.5, (b - 1940) / 10 - 0.5, color=BLUE, alpha=0.10)

    ax.set_xticks(x); ax.set_xticklabels([f"{int(d)}s" for d in counts.index])
    ax.set_ylabel("milestones in this timeline")
    ax.set_title("The pendulum, counted: which family was winning?")
    ax.legend(frameon=False, fontsize=8)
    if created:
        plt.tight_layout()
    return ax.figure


def fig_ingredients(ax=None):
    """When each of the three supplies arrived, and the gap before assembly."""
    ing = ingredients_available()
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(9.4, 2.9))

    rows = ["architecture", "compute", "data"]
    colours = {"architecture": SLATE, "compute": AMBER, "data": BLUE}
    for i, k in enumerate(rows):
        start = SUPPLY[k][0]
        ax.barh(i, 2026 - start, left=start, height=0.5, color=colours[k], alpha=0.85)
        ax.text(start + 1, i, f"  {k}: from {start}", va="center", fontsize=8.5,
                color="white", fontweight="bold")

    first = int(ing[ing.all_three].year.min())
    ax.axvline(first, color=GREEN, lw=2, ls="--")
    ax.text(first, 2.75, f" all three available\n {first}", color=GREEN, fontsize=8,
            fontweight="bold", va="top")
    ax.axvline(FIRST_COMBINATION[0], color=RED, lw=2)
    ax.text(FIRST_COMBINATION[0] + 0.6, -0.85,
            f"{FIRST_COMBINATION[1]} combines them, {FIRST_COMBINATION[0]}",
            color=RED, fontsize=8, fontweight="bold")

    ax.set_yticks([]); ax.set_xlim(1950, 2026); ax.set_ylim(-1.1, 2.9)
    ax.set_title("Availability is not assembly: the three-year gap of 2009 to 2012")
    if created:
        plt.tight_layout()
    return ax.figure


def fig_era_bands(ax=None):
    """The master timeline of booklet Chapter 15, as a band chart."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(10, 3.4))

    for i, (a, b, name, gloss) in enumerate(ERAS):
        winter = "Winter" in name
        ax.barh(0, b - a, left=a, height=0.9,
                color=BLUE if winter else (AMBER if i >= 6 else SLATE),
                alpha=0.30 if winter else 0.85, edgecolor="white")
        ax.text((a + b) / 2, 0.16, name, ha="center", fontsize=8,
                color=BLUE if winter else "white", fontweight="bold")
        ax.text((a + b) / 2, -0.22, gloss, ha="center", fontsize=6.4,
                color="#22293A" if winter else "white", wrap=True)

    ax.set_ylim(-0.7, 0.7); ax.set_xlim(1943, 2027); ax.set_yticks([])
    ax.set_title("Nine eras, two winters, one open ending")
    if created:
        plt.tight_layout()
    return ax.figure


def interactive_timeline():
    """ipywidgets year scrubber for the lab. Falls back to the static figure."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets not installed; showing the static wave instead.")
        return fig_hype_wave()

    df = load()
    slider = widgets.IntSlider(value=2026, min=1956, max=2026, step=1,
                               description="year", continuous_update=False,
                               layout=widgets.Layout(width="80%"))
    out = widgets.Output()

    def redraw(_=None):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 3.6))
            fig_hype_wave(annotate=6, upto=max(slider.value, 1958), ax=ax)
            plt.show()
            recent = df[(df.year <= slider.value) & (df.year > slider.value - 6)]
            for _, r in recent.iterrows():
                print(f"  {r.year}  {r['name']}\n        {r.note}")

    slider.observe(redraw, names="value")
    redraw()
    display(widgets.VBox([slider, out]))


def main(outdir="figures"):
    out = Path(outdir); out.mkdir(exist_ok=True)
    fig_hype_wave().savefig(out / "01_hype_wave.png", bbox_inches="tight")
    fig_family_fortunes().savefig(out / "02_family_fortunes.png", bbox_inches="tight")
    fig_ingredients().savefig(out / "03_ingredients.png", bbox_inches="tight")
    fig_era_bands().savefig(out / "04_era_bands.png", bbox_inches="tight")
    df = load()
    print(f"{len(df)} milestones plotted, {df.year.min()}-{df.year.max()}")
    print(f"Figures written to {out.resolve()}")


if __name__ == "__main__":
    main()
