"""
datasets.py, a lending dataset with bias mechanisms you can switch on and off.

The Evolution of Machine Learning and AI · Week 2 Lab
South East Technological University

Why synthetic data for an ethics lab?

Because with real data you can only ever observe that a disparity exists. Here
you can switch each mechanism on and off *one at a time* and watch which
fairness metric responds. That turns Chapter 3's seven-stage pipeline from a
diagram into an experiment.

The ground rule of this dataset, and the thing that makes it useful:

    Both groups have IDENTICAL latent creditworthiness.

There is no difference in ability to repay. Every disparity you will measure is
manufactured by one of the four mechanisms below, each of which corresponds to
a numbered stage of the booklet's pipeline. If you find yourself explaining a
disparity by saying "well, group B is just riskier", check the generator: you
wrote it, and it isn't.

    MECHANISM          PIPELINE STAGE   BOOKLET
    sampling_bias      2. Data          Ch. 4  representation bias
    historical_bias    1. World         Ch. 4  historical bias
    proxy              2. Data          Ch. 5  proxy variables
    label_bias         3. Labels        Ch. 5  label bias

Usage
-----
    from datasets import make_lending_data, train_model

    df = make_lending_data(n=6000, seed=2)            # all mechanisms on
    clean = make_lending_data(n=6000, seed=2, historical_bias=False,
                              proxy=False, label_bias=False)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "FEATURES",
    "PROTECTED",
    "MECHANISMS",
    "make_lending_data",
    "train_model",
    "population_table",
    "proxy_recoverability",
]

# Columns a model is allowed to see. Note what is absent: `group`.
# This is "fairness through unawareness" (Chapter 5), and the lab exists partly
# to demonstrate that it does not work.
FEATURES = [
    "income",
    "savings",
    "employment_years",
    "existing_debt",
    "credit_history_months",
    "postcode_score",
    "age",
]

PROTECTED = "group"

MECHANISMS = {
    "sampling_bias": (
        "Group B is under-sampled (20% of rows rather than 50%), so it exerts "
        "little pressure on the loss function. Pipeline stage 2."
    ),
    "historical_bias": (
        "At identical creditworthiness, group B has lower recorded income and "
        "savings and shorter credit histories: unequal access to credit and "
        "employment, not unequal ability. Pipeline stage 1."
    ),
    "proxy": (
        "postcode_score is strongly associated with group membership. It is the "
        "route by which a model that never sees `group` can still use it. "
        "Pipeline stage 2."
    ),
    "label_bias": (
        "Some group B borrowers who did in fact repay are recorded as having "
        "defaulted, because arrears were recorded more aggressively for them. "
        "The 'ground truth' column is a record of institutional behaviour. "
        "Pipeline stage 3."
    ),
}


def make_lending_data(
    n: int = 6000,
    seed: int = 0,
    sampling_bias: bool = True,
    historical_bias: bool = True,
    proxy: bool = True,
    label_bias: bool = True,
) -> pd.DataFrame:
    """Generate a loan-repayment dataset.

    Returns a DataFrame with the columns in FEATURES, plus:

        group      protected attribute, "A" or "B"
        repaid     the RECORDED outcome. This is what you train on, and it is
                   what an audit of the real world would give you.
        repaid_true
                   the outcome that actually happened, before label bias was
                   applied. It does not exist in real life. It is provided so
                   that you can measure how badly the recorded label misleads
                   you, and it must NEVER be used as a feature or as the
                   training target.
        latent     the underlying creditworthiness. Also unavailable in real
                   life; provided for the same diagnostic reason.
    """
    rng = np.random.default_rng(seed)

    # ---- group membership --------------------------------------------------
    p_b = 0.20 if sampling_bias else 0.50
    group = rng.choice(["A", "B"], size=n, p=[1 - p_b, p_b])
    is_b = group == "B"

    # ---- latent creditworthiness: IDENTICAL across groups ------------------
    latent = rng.normal(0.0, 1.0, n)

    # ---- what actually happens, driven only by latent ----------------------
    p_repay = 1.0 / (1.0 + np.exp(-(0.30 + 1.35 * latent)))
    repaid_true = (rng.random(n) < p_repay).astype(int)

    # ---- observed features -------------------------------------------------
    # Stage 1: at the same latent creditworthiness, group B's recorded
    # circumstances are worse, because access is not equally distributed.
    hist = 1.0 if historical_bias else 0.0

    income = (
        32_000
        + 9_000 * latent
        - hist * is_b * 6_500
        + rng.normal(0, 5_000, n)
    ).clip(9_000, None)

    savings = (
        4_000
        + 3_200 * latent
        - hist * is_b * 2_400
        + rng.normal(0, 1_800, n)
    ).clip(0, None)

    employment_years = (
        6.0 + 2.6 * latent - hist * is_b * 1.6 + rng.normal(0, 1.8, n)
    ).clip(0, 45)

    existing_debt = (
        9_000 - 2_600 * latent + hist * is_b * 1_900 + rng.normal(0, 2_400, n)
    ).clip(0, None)

    # Thin files: shorter credit histories mean noisier evidence for group B.
    credit_history_months = (
        84 + 22 * latent - hist * is_b * 30 + rng.normal(0, 18, n)
    ).clip(0, 400)

    age = (38 + 6 * latent + rng.normal(0, 9, n)).clip(18, 80)

    # Stage 2: the proxy. postcode_score carries group membership almost
    # perfectly when the mechanism is on, and carries nothing when it is off.
    if proxy:
        postcode_score = (
            0.72 * (~is_b).astype(float)
            + 0.12 * (latent - latent.mean()) / latent.std()
            + rng.normal(0, 0.11, n)
        )
    else:
        postcode_score = 0.36 + 0.12 * (latent - latent.mean()) / latent.std() \
            + rng.normal(0, 0.11, n)
    postcode_score = np.clip(postcode_score, 0, 1)

    # ---- Stage 3: the recorded label ---------------------------------------
    repaid = repaid_true.copy()
    if label_bias:
        # 18% of group B borrowers who repaid are recorded as defaulting.
        flip = is_b & (repaid_true == 1) & (rng.random(n) < 0.18)
        repaid[flip] = 0

    df = pd.DataFrame({
        "income": income.round(0),
        "savings": savings.round(0),
        "employment_years": employment_years.round(2),
        "existing_debt": existing_debt.round(0),
        "credit_history_months": credit_history_months.round(0),
        "postcode_score": postcode_score.round(4),
        "age": age.round(1),
        "group": group,
        "repaid": repaid,
        "repaid_true": repaid_true,
        "latent": latent.round(4),
    })
    df.attrs["mechanisms"] = {
        "sampling_bias": sampling_bias,
        "historical_bias": historical_bias,
        "proxy": proxy,
        "label_bias": label_bias,
    }
    return df


# ----------------------------------------------------------------------------
# A model, trained the way a competent and well-meaning team would train it
# ----------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    target: str = "repaid",
    test_size: float = 0.35,
    seed: int = 0,
    model: str = "gbm",
):
    """Fit a model on FEATURES only. `group` is deliberately withheld.

    Returns (pipeline, train_df, test_df, scores_test).

    The scores are calibrated probabilities of repayment, so a high score means
    "approve". Nothing about this function is careless: it is exactly what a
    good team does, and the disparities you are about to measure survive it.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[[target, "group"]]
    )

    if model == "logreg":
        clf = Pipeline([
            ("scale", ColumnTransformer([("num", StandardScaler(), FEATURES)])),
            ("clf", LogisticRegression(max_iter=2000)),
        ])
    elif model == "gbm":
        clf = Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_iter=220, learning_rate=0.07, random_state=seed)),
        ])
    else:
        raise ValueError("model must be 'logreg' or 'gbm'")

    clf.fit(train_df[FEATURES], train_df[target])
    scores = clf.predict_proba(test_df[FEATURES])[:, 1]
    return clf, train_df.reset_index(drop=True), test_df.reset_index(drop=True), scores


# ----------------------------------------------------------------------------
# Stage 2 audit helpers (booklet Chapter 4)
# ----------------------------------------------------------------------------

def population_table(df: pd.DataFrame, target: str = "repaid") -> pd.DataFrame:
    """Counts and feature means by group: the first table of any audit.

    Chapter 4's instruction is to build this before touching a model. It takes
    one line and it is skipped more often than any other step in this booklet.
    """
    rows = []
    for g, sub in df.groupby("group"):
        row = {
            "group": g,
            "n": len(sub),
            "share": len(sub) / len(df),
            f"{target}_rate": sub[target].mean(),
        }
        for f in FEATURES:
            row[f"mean_{f}"] = sub[f].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def proxy_recoverability(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Try to predict the protected attribute from the permitted features.

    This is the ten-minute experiment recommended in booklet Chapter 5, and it
    is the most persuasive demonstration available to you when a stakeholder
    believes that dropping a column solved the problem. Returns per-feature
    importance so you can see *which* column is doing the leaking.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    X = df[FEATURES]
    a = (df["group"] == "B").astype(int)
    Xtr, Xte, atr, ate = train_test_split(X, a, test_size=0.35, random_state=seed, stratify=a)

    rf = RandomForestClassifier(n_estimators=250, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, atr)
    acc = rf.score(Xte, ate)
    majority = max(ate.mean(), 1 - ate.mean())

    imp = permutation_importance(rf, Xte, ate, n_repeats=6, random_state=seed, n_jobs=-1)
    out = pd.DataFrame({
        "feature": FEATURES,
        "importance": imp.importances_mean,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    out.attrs["accuracy"] = float(acc)
    out.attrs["majority_baseline"] = float(majority)
    return out


if __name__ == "__main__":
    df = make_lending_data(n=6000, seed=1)
    print("mechanisms:", df.attrs["mechanisms"], "\n")
    print(population_table(df).round(3).to_string(index=False), "\n")
    rec = proxy_recoverability(df)
    print(f"Predicting `group` from the permitted features alone:")
    print(f"  accuracy         {rec.attrs['accuracy']:.3f}")
    print(f"  majority baseline{rec.attrs['majority_baseline']:>7.3f}\n")
    print(rec.round(4).to_string(index=False))
