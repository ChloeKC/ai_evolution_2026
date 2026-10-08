"""
explain.py, SHAP and LIME, plus a from-scratch LIME so the lab never blocks.

The Evolution of Machine Learning and AI · Week 2 Lab
South East Technological University

Booklet cross-reference: Chapter 12, "Explainability, and What It Cannot Do".

Three things this module is for:

 1. Catching a proxy. This is what local explanation is genuinely excellent at,
    and it is how a great many real fairness problems are actually found.
 2. Demonstrating the DISAGREEMENT PROBLEM. Run two methods on the same case
    and compare the orderings. If they disagree, that is a finding, not noise.
 3. Making the limits concrete: an explanation reports what the model USED. It
    has no opinion about whether the model SHOULD have.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from datasets import FEATURES

SLATE, AMBER, RED, GREEN, GREY = "#176B78", "#C98200", "#B63D32", "#1E6F52", "#6F858A"


# ---------------------------------------------------------------------------
# A LIME implementation in 30 lines, so you can see there is no magic
# ---------------------------------------------------------------------------

def lime_from_scratch(model, X_background: pd.DataFrame, x: pd.Series,
                      n_samples: int = 4000, kernel_width: float | None = None,
                      seed: int = 0) -> pd.Series:
    """Explain one prediction by fitting a weighted local linear surrogate.

    The whole method: perturb the instance, ask the model what it now says,
    weight the perturbations by how near they are to the original, and fit a
    ridge regression. The coefficients are the explanation.

    Note the two free choices buried in here: the perturbation distribution and
    the kernel width. Change either and the explanation changes. That is the
    origin of LIME's neighbourhood sensitivity, discussed in Chapter 12.
    """
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    cols = list(X_background.columns)
    mu = np.array(X_background[cols].mean().to_numpy(), dtype=float, copy=True)
    sd = np.array(X_background[cols].std().to_numpy(), dtype=float, copy=True)
    sd[sd == 0] = 1.0

    x_vec = x[cols].to_numpy(dtype=float)
    z_std = rng.normal(0, 1, size=(n_samples, len(cols)))
    Z = mu + z_std * sd                      # samples in the data's own scale
    Z[0] = x_vec                             # keep the instance itself

    preds = model.predict_proba(pd.DataFrame(Z, columns=cols))[:, 1]

    d = np.sqrt(np.mean(((Z - x_vec) / sd) ** 2, axis=1))
    if kernel_width is None:
        kernel_width = np.sqrt(len(cols)) * 0.75
    w = np.exp(-(d ** 2) / (kernel_width ** 2))

    ridge = Ridge(alpha=1.0)
    ridge.fit((Z - mu) / sd, preds, sample_weight=w)
    return pd.Series(ridge.coef_, index=cols).sort_values(key=np.abs, ascending=False)


# ---------------------------------------------------------------------------
# Library-backed explanations, each degrading gracefully
# ---------------------------------------------------------------------------

def shap_values(model, X_background: pd.DataFrame, X_explain: pd.DataFrame,
                max_background: int = 200):
    """SHAP values for a scikit-learn pipeline. Returns a DataFrame or None."""
    try:
        import shap
    except ImportError:
        print("shap not installed  ->  pip install shap")
        return None

    bg = shap.sample(X_background, min(max_background, len(X_background)), random_state=0)
    f = lambda data: model.predict_proba(pd.DataFrame(data, columns=list(X_background.columns)))[:, 1]
    explainer = shap.KernelExplainer(f, bg)
    vals = explainer.shap_values(X_explain, nsamples=200, silent=True)
    out = pd.DataFrame(np.asarray(vals), columns=list(X_explain.columns),
                       index=X_explain.index)
    out.attrs["base_value"] = float(np.asarray(explainer.expected_value).ravel()[0])
    return out


def lime_library(model, X_background: pd.DataFrame, x: pd.Series, n_samples: int = 4000):
    """The reference LIME implementation, for comparison with ours."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print("lime not installed  ->  pip install lime")
        return None

    cols = list(X_background.columns)
    expl = LimeTabularExplainer(
        X_background[cols].to_numpy(), feature_names=cols,
        class_names=["default", "repaid"], discretize_continuous=False,
        random_state=0,
    )
    e = expl.explain_instance(
        x[cols].to_numpy(),
        lambda data: model.predict_proba(pd.DataFrame(data, columns=cols)),
        num_features=len(cols), num_samples=n_samples,
    )
    return pd.Series(dict(e.as_list())).reindex(cols)


# ---------------------------------------------------------------------------
# The disagreement problem, made measurable
# ---------------------------------------------------------------------------

def compare_explanations(model, X_background: pd.DataFrame, x: pd.Series,
                         shap_row: pd.Series | None = None) -> pd.DataFrame:
    """Put every available method's ranking side by side for one instance."""
    cols = list(X_background.columns)
    table = pd.DataFrame(index=cols)

    ours = lime_from_scratch(model, X_background, x)
    table["lime_ours"] = ours.reindex(cols)

    lib = lime_library(model, X_background, x)
    if lib is not None:
        table["lime_library"] = lib

    if shap_row is not None:
        table["shap"] = shap_row.reindex(cols)

    for c in list(table.columns):
        table[f"rank_{c}"] = table[c].abs().rank(ascending=False).astype("Int64")

    rank_cols = [c for c in table.columns if c.startswith("rank_")]
    if len(rank_cols) > 1:
        from scipy.stats import spearmanr
        m = spearmanr(table[rank_cols].to_numpy()).statistic
        m = np.atleast_2d(m)
        iu = np.triu_indices_from(m, k=1)
        table.attrs["rank_correlation"] = float(np.mean(m[iu]))
        table.attrs["rank_correlation_matrix"] = pd.DataFrame(
            m, index=rank_cols, columns=rank_cols)
    return table.sort_values("lime_ours", key=np.abs, ascending=False)


def global_importance(model, X: pd.DataFrame, y: pd.Series, seed: int = 0) -> pd.DataFrame:
    """Global feature importance by permutation. Complements the local view.

    Local explanations tell you about one person. This tells you what the model
    leans on across everybody, which is where a proxy shows up most clearly.
    """
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, n_repeats=8, random_state=seed,
                               n_jobs=-1, scoring="roc_auc")
    return (pd.DataFrame({"feature": list(X.columns),
                          "importance": r.importances_mean,
                          "std": r.importances_std})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def fig_local_explanation(contrib: pd.Series, title: str = "Why was this applicant refused?",
                          highlight: str = "postcode_score"):
    """The waterfall-style bar chart used on the lecture slide."""
    import matplotlib.pyplot as plt

    c = contrib.sort_values(key=np.abs, ascending=True)
    colours = [GREEN if v > 0 else RED for v in c.values]
    alphas = [1.0 if i == highlight else 0.55 for i in c.index]

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for (name, v), col, al in zip(c.items(), colours, alphas):
        ax.barh(name, v, color=col, alpha=al)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel("← pushes toward REFUSE          contribution          pushes toward APPROVE →")
    ax.set_title(title)
    if highlight in c.index:
        ax.get_yticklabels()[list(c.index).index(highlight)].set_fontweight("bold")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    from datasets import make_lending_data, train_model

    df = make_lending_data(n=6000, seed=1)
    model, train, test, scores = train_model(df, seed=1)

    # Pick a refused applicant from the under-served group.
    refused = test[(scores < 0.4) & (test.group == "B")].index[0]
    x = test.loc[refused, FEATURES]

    print("Explaining one refused applicant from group B\n")
    ours = lime_from_scratch(model, train[FEATURES], x)
    print("LIME (from scratch):")
    print(ours.round(4).to_string(), "\n")

    sv = shap_values(model, train[FEATURES], test.loc[[refused], FEATURES])
    shap_row = sv.iloc[0] if sv is not None else None
    if shap_row is not None:
        print("SHAP:")
        print(shap_row.sort_values(key=np.abs, ascending=False).round(4).to_string(), "\n")

    cmp = compare_explanations(model, train[FEATURES], x, shap_row)
    print("Side by side (NOTE: LIME coefficients are local SLOPES, SHAP values")
    print("are ATTRIBUTIONS of this instance's actual values. Compare the")
    print("RANKINGS, not the signs.):")
    print(cmp[[c for c in cmp.columns if not c.startswith("rank_")]].round(4).to_string())
    if "rank_correlation" in cmp.attrs:
        print(f"\nMean Spearman rank correlation between methods: "
              f"{cmp.attrs['rank_correlation']:.3f}")
        print("Well below 1.0 is the disagreement problem of Chapter 12: two")
        print("defensible methods, one applicant, different stories.")

    print("\n" + "=" * 62)
    print("GLOBAL view: what does the model lean on across everybody?")
    gi = global_importance(model, test[FEATURES], test.repaid)
    print(gi.round(4).to_string(index=False))
    from datasets import proxy_recoverability
    rec = proxy_recoverability(df)
    print(f"\n  `group` is recoverable from the permitted features with "
          f"{rec.attrs['accuracy']:.1%} accuracy")
    print(f"  (majority baseline {rec.attrs['majority_baseline']:.1%}).")
    print("\n  Note what the global importances show, and do not show. The most")
    print("  important features for PREDICTING REPAYMENT are income, savings and")
    print("  credit history, not the obvious geographic proxy. Yet the model")
    print("  reproduces the group disparity anyway, because historical bias made")
    print("  income, savings and credit history into proxies too.")
    print("\n  This is the uncomfortable finding of the lab: there is no single")
    print("  column to delete. Run mitigate.py to watch dropping postcode_score")
    print("  fail to fix anything. See booklet Chapter 5.")
