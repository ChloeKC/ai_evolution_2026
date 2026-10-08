"""
mitigate.py, three families of mitigation, and one instructive failure.

The Evolution of Machine Learning and AI · Week 2 Lab
South East Technological University

Booklet cross-reference: Chapter 15, "The Practice", and Chapter 7 on why the
earliest fix is the cheapest.

    drop_the_proxy()        the intuitive fix. Watch it not work.
    reweight()              PRE-processing: change the data's influence
    worst_group_threshold() IN-processing (approximated): optimise the worst group
    per_group_thresholds()  POST-processing: the effective and legally exposed one
    fix_the_label()         STAGE 3: the fix nobody reaches for, and the best one
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fairness as fm
from datasets import FEATURES, make_lending_data, train_model


def _report(name, y, s, a, t, extra=None):
    df = fm.metrics_by_group(y, s, a, t)
    per = df[df.group != "ALL"]
    row = {
        "strategy": name,
        "accuracy": float(df[df.group == "ALL"].accuracy.iloc[0]),
        "selection_gap": float(per.selection_rate.max() - per.selection_rate.min()),
        "tpr_gap": float(per.tpr.max() - per.tpr.min()),
        "fpr_gap": float(per.fpr.max() - per.fpr.min()),
        "ppv_gap": float(per.ppv.max() - per.ppv.min()),
    }
    if extra:
        row.update(extra)
    return row


def drop_the_proxy(df, proxy_col="postcode_score", seed=1):
    """Remove the obvious proxy column and re-audit.

    This is the first thing every team tries. In this dataset it removes almost
    none of the disparity, because historical bias (stage 1) has already made
    income, savings and credit history carry group membership as well. The
    lesson generalises: proxy leakage is usually distributed, so there is no
    single column whose deletion fixes the outcome.
    """
    import datasets
    original = list(datasets.FEATURES)
    try:
        datasets.FEATURES = [f for f in original if f != proxy_col]
        _, _, test, s = train_model(df, seed=seed)
        return _report(f"drop {proxy_col}", test.repaid, s, test.group, 0.5), test, s
    finally:
        datasets.FEATURES = original


def reweight(df, seed=1):
    """PRE-processing. Reweight training rows so each (group, outcome) cell
    carries the influence it would have had under independence."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df, test_size=0.35, random_state=seed,
                                   stratify=df[["repaid", "group"]])
    n = len(train)
    w = np.ones(n)
    for g in train.group.unique():
        for yv in (0, 1):
            m = ((train.group == g) & (train.repaid == yv)).to_numpy()
            if m.sum() == 0:
                continue
            expected = (train.group == g).mean() * (train.repaid == yv).mean()
            w[m] = expected / (m.sum() / n)

    clf = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.07,
                                         random_state=seed)
    clf.fit(train[FEATURES], train.repaid, sample_weight=w)
    s = clf.predict_proba(test[FEATURES])[:, 1]
    return _report("reweighting (pre)", test.repaid, s, test.group, 0.5), test, s


def per_group_thresholds(y, s, a, criterion="selection_gap", grid=None):
    """POST-processing. Search per-group thresholds that minimise one gap.

    Effective, cheap, works on a model you cannot retrain, and in the EU it
    uses the protected attribute at decision time, which moves you toward
    direct discrimination. Chapter 7. Never deploy this without legal advice.
    """
    a = np.asarray(a)
    groups = sorted(pd.unique(a))
    grid = np.linspace(0.10, 0.90, 41) if grid is None else grid
    best, best_val = None, np.inf
    for t0 in grid:
        for t1 in grid:
            thr = {groups[0]: float(t0), groups[1]: float(t1)}
            r = _report("x", y, s, a, thr)
            score = r[criterion] - 0.05 * r["accuracy"]
            if score < best_val:
                best_val, best, best_row = score, thr, r
    best_row["strategy"] = f"per-group thresholds (post, {criterion})"
    best_row["thresholds"] = {g: round(v, 3) for g, v in best.items()}
    return best_row, best


def fix_the_label(df_biased_seed=1, n=8000):
    """STAGE 3. Retrain on the outcome that actually happened.

    In the real world you cannot do this, because the unbiased label does not
    exist: that is the entire problem. Here it does exist, so you can measure
    exactly how much of the disparity was manufactured by the label rather than
    by the model. It is usually far more than teams expect, and it is why
    Chapter 6 insists that the cheapest fix is always the earliest one.
    """
    df = make_lending_data(n=n, seed=df_biased_seed)
    _, _, test, s = train_model(df, target="repaid_true", seed=df_biased_seed)
    return _report("fix the label (stage 3)", test.repaid_true, s, test.group, 0.5), test, s


def compare_all(n=8000, seed=1) -> pd.DataFrame:
    df = make_lending_data(n=n, seed=seed)
    _, _, test, s = train_model(df, seed=seed)
    rows = [_report("baseline (nothing done)", test.repaid, s, test.group, 0.5)]

    r, _, _ = drop_the_proxy(df, seed=seed); rows.append(r)
    r, _, _ = reweight(df, seed=seed); rows.append(r)
    r, _ = per_group_thresholds(test.repaid, s, test.group, "selection_gap"); rows.append(r)
    r, _ = per_group_thresholds(test.repaid, s, test.group, "fpr_gap"); rows.append(r)
    r, _, _ = fix_the_label(seed, n); rows.append(r)

    out = pd.DataFrame(rows)
    cols = ["strategy", "accuracy", "selection_gap", "tpr_gap", "fpr_gap", "ppv_gap"]
    return out[cols + [c for c in out.columns if c not in cols]]


if __name__ == "__main__":
    tbl = compare_all()
    pd.set_option("display.width", 160)
    print("Mitigations compared, all at t = 0.5 unless stated\n")
    print(tbl.round(4).to_string(index=False))
    print("""
Read the table in this order:

  1. Dropping the obvious proxy does NOT help, and here it makes the gap
     WORSE: the model leans harder on the remaining proxies. There is no
     single column to delete.
  2. Reweighting helps a little and costs a little accuracy.
  3. Per-group thresholds are the most effective post-hoc lever, and each
     configuration only fixes the ONE gap it was tuned for (Chapter 9).
  4. Fixing the label does more than all the model-side work combined,
     because that is where the harm was actually introduced.

The cheapest real fix is always the earliest one. Booklet Chapter 15.""")
