"""
fairness.py, the four families of fairness metric, implemented from scratch.

The Evolution of Machine Learning and AI · Week 2 Lab
South East Technological University

Everything here is deliberately written out in numpy rather than imported from
a fairness library. The point of the Week 2 lab is that these quantities are
*arithmetic on a confusion matrix*, not magic supplied by a package. Once you
have seen how short they are, you will never again accept "the tool says the
model is fair" as an answer.

Booklet cross-references:
    Chapter 7 , the threshold and the confusion matrix read morally
    Chapter 8 , the four families of fairness definition
    Chapter 9 , the impossibility result

Notation used throughout (matching the booklet):
    y      true outcome, 1 = positive outcome (e.g. repaid the loan)
    s      model score in [0, 1]
    a      protected attribute (group label)
    t      decision threshold; yhat = (s >= t)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "GroupMetrics",
    "confusion",
    "group_metrics",
    "metrics_by_group",
    "demographic_parity_difference",
    "equal_opportunity_difference",
    "equalised_odds_difference",
    "predictive_parity_difference",
    "calibration_curve_by_group",
    "calibration_error_by_group",
    "threshold_sweep",
    "summarise",
]


# ----------------------------------------------------------------------------
# Confusion matrix, and everything that falls out of it
# ----------------------------------------------------------------------------

@dataclass
class GroupMetrics:
    """Every rate a fairness criterion in Chapter 8 needs, for one group."""

    group: str
    n: int
    base_rate: float          # P(Y=1)            how often the outcome truly occurs
    selection_rate: float     # P(Yhat=1)         how often we say yes
    accuracy: float
    tpr: float                # P(Yhat=1 | Y=1)   true positive rate / recall
    fpr: float                # P(Yhat=1 | Y=0)   false positive rate
    fnr: float                # P(Yhat=0 | Y=1)   false negative rate
    tnr: float                # P(Yhat=0 | Y=0)   true negative rate
    ppv: float                # P(Y=1 | Yhat=1)   precision / predictive parity
    npv: float                # P(Y=0 | Yhat=0)
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> dict:
        return asdict(self)


def _safe_div(numerator: float, denominator: float) -> float:
    """Rates are undefined when the denominator is empty; say so with nan.

    Returning 0.0 here would be a lie that silently flatters the model: a group
    with no positive cases would appear to have a perfect false-negative rate.
    """
    return float("nan") if denominator == 0 else numerator / denominator


def confusion(y: np.ndarray, yhat: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) as plain integers."""
    y = np.asarray(y).astype(bool)
    yhat = np.asarray(yhat).astype(bool)
    tp = int(np.sum(y & yhat))
    fp = int(np.sum(~y & yhat))
    fn = int(np.sum(y & ~yhat))
    tn = int(np.sum(~y & ~yhat))
    return tp, fp, fn, tn


def group_metrics(y: np.ndarray, s: np.ndarray, t: float, label: str = "all") -> GroupMetrics:
    """All rates for a single group at threshold t."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    yhat = (s >= t).astype(int)
    tp, fp, fn, tn = confusion(y, yhat)
    n = len(y)
    return GroupMetrics(
        group=label,
        n=n,
        base_rate=_safe_div(tp + fn, n),
        selection_rate=_safe_div(tp + fp, n),
        accuracy=_safe_div(tp + tn, n),
        tpr=_safe_div(tp, tp + fn),
        fpr=_safe_div(fp, fp + tn),
        fnr=_safe_div(fn, tp + fn),
        tnr=_safe_div(tn, fp + tn),
        ppv=_safe_div(tp, tp + fp),
        npv=_safe_div(tn, tn + fn),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def metrics_by_group(
    y: np.ndarray,
    s: np.ndarray,
    a: Sequence,
    t: float | dict = 0.5,
) -> pd.DataFrame:
    """One row per group, plus an 'ALL' row.

    `t` may be a single float (one rule for everybody) or a dict mapping group
    label -> threshold (per-group thresholds; see Chapter 7 on why this is
    legally exposed in the EU even when it is statistically convenient).
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    a = np.asarray(a)

    rows = []
    for g in sorted(pd.unique(a)):
        mask = a == g
        tg = t[g] if isinstance(t, dict) else t
        m = group_metrics(y[mask], s[mask], tg, label=str(g))
        row = m.as_dict()
        row["threshold"] = tg
        rows.append(row)

    if isinstance(t, dict):
        yhat_all = np.zeros(len(y), dtype=int)
        for g, tg in t.items():
            yhat_all[a == g] = (s[a == g] >= tg).astype(int)
        tp, fp, fn, tn = confusion(y, yhat_all)
        overall = GroupMetrics(
            group="ALL", n=len(y),
            base_rate=_safe_div(tp + fn, len(y)),
            selection_rate=_safe_div(tp + fp, len(y)),
            accuracy=_safe_div(tp + tn, len(y)),
            tpr=_safe_div(tp, tp + fn), fpr=_safe_div(fp, fp + tn),
            fnr=_safe_div(fn, tp + fn), tnr=_safe_div(tn, fp + tn),
            ppv=_safe_div(tp, tp + fp), npv=_safe_div(tn, tn + fn),
            tp=tp, fp=fp, fn=fn, tn=tn,
        ).as_dict()
        overall["threshold"] = float("nan")
    else:
        overall = group_metrics(y, s, t, label="ALL").as_dict()
        overall["threshold"] = t
    rows.append(overall)

    cols = ["group", "n", "threshold", "base_rate", "selection_rate", "accuracy",
            "tpr", "fpr", "fnr", "ppv", "tp", "fp", "fn", "tn"]
    return pd.DataFrame(rows)[cols]


# ----------------------------------------------------------------------------
# The four families, as single numbers (max gap between any two groups)
# ----------------------------------------------------------------------------

def _max_gap(values: Iterable[float]) -> float:
    vals = [v for v in values if not np.isnan(v)]
    return float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")


def demographic_parity_difference(y, s, a, t=0.5) -> float:
    """Family 1. Largest gap in selection rate between groups. 0 = parity.

    Note this never looks at `y`. Demographic parity is entirely a statement
    about who we selected, not about whether we were right.
    """
    df = metrics_by_group(y, s, a, t)
    df = df[df.group != "ALL"]
    return _max_gap(df.selection_rate)


def equal_opportunity_difference(y, s, a, t=0.5) -> float:
    """Family 2 (weak form). Largest gap in true-positive rate."""
    df = metrics_by_group(y, s, a, t)
    df = df[df.group != "ALL"]
    return _max_gap(df.tpr)


def equalised_odds_difference(y, s, a, t=0.5) -> float:
    """Family 2 (strong form). Worst of the TPR gap and the FPR gap."""
    df = metrics_by_group(y, s, a, t)
    df = df[df.group != "ALL"]
    return float(np.nanmax([_max_gap(df.tpr), _max_gap(df.fpr)]))


def predictive_parity_difference(y, s, a, t=0.5) -> float:
    """Family 3 (thresholded). Largest gap in positive predictive value."""
    df = metrics_by_group(y, s, a, t)
    df = df[df.group != "ALL"]
    return _max_gap(df.ppv)


# ----------------------------------------------------------------------------
# Calibration: the criterion that has nothing to do with a threshold
# ----------------------------------------------------------------------------

def calibration_curve_by_group(y, s, a, n_bins: int = 10) -> pd.DataFrame:
    """Observed outcome frequency against predicted score, per group.

    A perfectly calibrated score lies on the diagonal *within every group*:
    among everyone scored 0.7, about 70% experience the outcome. This is the
    criterion vendors of risk scores defend (Chapter 9, the COMPAS dispute),
    and it is the one that collides with equal error rates.
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    a = np.asarray(a)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    for g in sorted(pd.unique(a)):
        m = a == g
        idx = np.clip(np.digitize(s[m], edges[1:-1]), 0, n_bins - 1)
        for b in range(n_bins):
            sel = idx == b
            if sel.sum() == 0:
                continue
            rows.append({
                "group": str(g),
                "bin": b,
                "bin_centre": float((edges[b] + edges[b + 1]) / 2),
                "n": int(sel.sum()),
                "mean_score": float(s[m][sel].mean()),
                "observed_rate": float(y[m][sel].mean()),
            })
    return pd.DataFrame(rows)


def calibration_error_by_group(y, s, a, n_bins: int = 10) -> pd.DataFrame:
    """Expected calibration error (ECE): mean |predicted - observed|, weighted."""
    cc = calibration_curve_by_group(y, s, a, n_bins)
    out = []
    for g, sub in cc.groupby("group"):
        w = sub.n / sub.n.sum()
        ece = float(np.sum(w * np.abs(sub.mean_score - sub.observed_rate)))
        out.append({"group": g, "ece": ece, "n": int(sub.n.sum())})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# The threshold sweep: the heart of the Bias & Fairness Explorer
# ----------------------------------------------------------------------------

def threshold_sweep(y, s, a, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Recompute every criterion across the whole range of thresholds.

    This single table is the lab's central artefact. Plot any two columns
    against each other and you are looking at a trade-off that somebody has to
    decide, and that nobody can eliminate (Chapter 9).
    """
    if thresholds is None:
        thresholds = np.linspace(0.02, 0.98, 97)

    rows = []
    for t in thresholds:
        per_group = metrics_by_group(y, s, a, float(t))
        overall = per_group[per_group.group == "ALL"].iloc[0]
        rows.append({
            "threshold": float(t),
            "accuracy": float(overall.accuracy),
            "selection_rate": float(overall.selection_rate),
            "demographic_parity_diff": demographic_parity_difference(y, s, a, float(t)),
            "equal_opportunity_diff": equal_opportunity_difference(y, s, a, float(t)),
            "equalised_odds_diff": equalised_odds_difference(y, s, a, float(t)),
            "predictive_parity_diff": predictive_parity_difference(y, s, a, float(t)),
        })
    return pd.DataFrame(rows)


def summarise(y, s, a, t=0.5) -> str:
    """A short human-readable audit paragraph, suitable for a model card."""
    df = metrics_by_group(y, s, a, t)
    per = df[df.group != "ALL"]
    lines = [f"Threshold t = {t:.2f}   (n = {int(df[df.group == 'ALL'].n.iloc[0])})", ""]
    lines.append(f"{'group':>8} {'n':>7} {'base':>7} {'select':>7} {'acc':>7} "
                 f"{'TPR':>7} {'FPR':>7} {'PPV':>7}")
    for _, r in df.iterrows():
        lines.append(f"{r.group:>8} {int(r.n):>7} {r.base_rate:>7.3f} "
                     f"{r.selection_rate:>7.3f} {r.accuracy:>7.3f} "
                     f"{r.tpr:>7.3f} {r.fpr:>7.3f} {r.ppv:>7.3f}")
    lines += [
        "",
        f"  demographic parity gap (selection rate) : {_max_gap(per.selection_rate):.3f}",
        f"  equal opportunity gap  (TPR)            : {_max_gap(per.tpr):.3f}",
        f"  false-positive rate gap                 : {_max_gap(per.fpr):.3f}",
        f"  predictive parity gap  (PPV)            : {_max_gap(per.ppv):.3f}",
        "",
        "A gap of 0.000 on every line at once is not achievable when base rates",
        "differ and the classifier is imperfect. See booklet Chapter 9.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Tiny self-test with a deliberately unfair score.
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.choice(["A", "B"], size=n, p=[0.7, 0.3])
    base = np.where(a == "A", 0.62, 0.45)
    y = (rng.random(n) < base).astype(int)
    s = np.clip(base + 0.25 * y + rng.normal(0, 0.16, n), 0, 1)
    print(summarise(y, s, a, 0.5))
