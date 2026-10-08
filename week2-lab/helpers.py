"""helpers.py, tiny wrappers so that each lab step is ONE cell.

The Evolution of Machine Learning and AI · Week 2 Lab (easy edition) · SETU

You never need to edit this file. Every function prints a small, readable
result and returns it as a DataFrame in case you want to look closer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fairness as fm
from datasets import FEATURES, make_lending_data, train_model, population_table

pd.set_option("display.width", 200)


import json
from pathlib import Path

DATA_CSV = Path(__file__).with_name("data") / "lending_data.csv"
DATA_JSON = Path(__file__).with_name("data") / "lending_data.json"


def load_dataset() -> pd.DataFrame:
    """Read data/lending_data.csv (8,000 applicants). Falls back to the generator if missing."""
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        meta = json.load(open(DATA_JSON)) if DATA_JSON.exists() else {}
        df.attrs["mechanisms"] = meta.get("mechanisms", {
            "sampling_bias": True, "historical_bias": True, "proxy": True, "label_bias": True})
        df.attrs["source"] = str(DATA_CSV)
        return df
    df = make_lending_data(n=8000, seed=1)
    df.attrs["source"] = "generated (data/lending_data.csv not found)"
    return df


def load_everything(n: int = 8000, seed: int = 1):
    """Load the dataset file, train the standard model, return the pieces the lab uses.

    Returns a dict with keys: df, model, train, test, scores, y, group.
    """
    df = load_dataset()
    model, train, test, scores = train_model(df, seed=seed)
    return {
        "df": df, "model": model, "train": train, "test": test, "scores": scores,
        "y": test.repaid.to_numpy(), "group": test.group.to_numpy(),
    }


def try_threshold(lab: dict, t: float) -> pd.DataFrame:
    """Apply ONE approval threshold to everybody and show the four numbers per group."""
    y, s, a = lab["y"], lab["scores"], lab["group"]
    full = fm.metrics_by_group(y, s, a, t)
    tbl = full[["group", "n", "selection_rate", "accuracy", "tpr", "fpr"]].round(3)
    tbl.columns = ["group", "n", "approval rate", "accuracy", "TPR (good borrowers approved)", "FPR (bad borrowers approved)"]
    gap = fm.demographic_parity_difference(y, s, a, t)
    sel_all = float(full.loc[full.group == "ALL", "selection_rate"].iloc[0])
    print(f"threshold {t:.2f}   approval rate overall {sel_all:.1%}   "
          f"gap in approval rate between groups {gap:.3f}")
    return tbl


def find_equal_approval_threshold(lab: dict) -> pd.DataFrame:
    """Scan thresholds and show the five where the approval-rate gap is smallest."""
    sweep = fm.threshold_sweep(lab["y"], lab["scores"], lab["group"])
    best = sweep.nsmallest(5, "demographic_parity_diff")[
        ["threshold", "demographic_parity_diff", "selection_rate", "accuracy"]].round(3)
    best.columns = ["threshold", "gap in approval rate", "approval rate overall", "accuracy"]
    return best.reset_index(drop=True)


def mechanism_table(n: int = 8000, seed: int = 1) -> pd.DataFrame:
    """Switch on ONE bias mechanism at a time and report the approval-rate gap.

    This is the old Exercise 1, done for you in one cell instead of five hours.
    """
    switches = ["sampling_bias", "historical_bias", "proxy", "label_bias"]
    rows = []
    for on in [None] + switches + ["all four"]:
        kw = {s: False for s in switches}
        if on == "all four":
            kw = {s: True for s in switches}
        elif on is not None:
            kw[on] = True
        df = make_lending_data(n=n, seed=seed, **kw)
        model, train, test, scores = train_model(df, seed=seed)
        y, a = test.repaid.to_numpy(), test.group.to_numpy()
        rows.append({
            "mechanism switched on": on or "none (clean data)",
            "gap in approval rate": round(fm.demographic_parity_difference(y, scores, a, 0.5), 3),
            "gap in TPR (equal opportunity)": round(fm.equal_opportunity_difference(y, scores, a, 0.5), 3),
            "accuracy": round(float(((scores >= 0.5).astype(int) == y).mean()), 3),
        })
    return pd.DataFrame(rows)


def worst_group_accuracy(lab: dict, model_name: str = "gbm", seed: int = 1) -> pd.DataFrame:
    """Train the named model ('gbm' or 'logreg') and report accuracy per group."""
    _, _, test, scores = train_model(lab["df"], seed=seed, model=model_name)
    y, a = test.repaid.to_numpy(), test.group.to_numpy()
    tbl = fm.metrics_by_group(y, scores, a, 0.5)[["group", "accuracy"]].round(3)
    tbl.insert(0, "model", model_name)
    return tbl


MECHANISMS = {
    "sampling_bias":   ("Data",   "group B is only about 20% of the file, not 50%"),
    "historical_bias": ("World",  "at equal ability, B has lower recorded income and savings and a shorter credit history"),
    "proxy":           ("Data",   "postcode_score almost perfectly reveals which group someone is in"),
    "label_bias":      ("Labels", "some group B people who repaid were recorded as defaulting"),
}


def describe_mechanisms(df: pd.DataFrame) -> None:
    """Print the four bias mechanisms, which pipeline stage each belongs to, and what it did."""
    on = df.attrs.get("mechanisms", {})
    print("The four bias mechanisms in this dataset (lecture: the pipeline picture)\n")
    print(f"  {'':3} {'mechanism':<17} {'pipeline stage':<15} what it did to the data")
    for name, (stage, what) in MECHANISMS.items():
        mark = "[x]" if on.get(name, True) else "[ ]"
        print(f"  {mark} {name:<17} {stage:<15} {what}")
    print("\n  Stage names come from the lecture: World -> Data -> Labels -> Objective -> Model -> Threshold -> Deploy")
