"""
run_lab.py, the whole Week 2 lab, end to end, as one script.

    python run_lab.py

Produces every figure and table in ./figures and prints the audit narrative.
Use this to check your environment before the lab, or to regenerate the
teaching figures. The notebook walks the same path with commentary.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import fairness as fm
import explorer as ex
from datasets import (FEATURES, MECHANISMS, make_lending_data, population_table,
                      proxy_recoverability, train_model)


def rule(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main():
    out = Path("figures"); out.mkdir(exist_ok=True)
    pd.set_option("display.width", 170)

    rule("STAGE 1-3 · THE DATASET, AND WHAT WAS DONE TO IT")
    df = make_lending_data(n=8000, seed=1)
    for k, on in df.attrs["mechanisms"].items():
        print(f"  [{'x' if on else ' '}] {k:16s} {MECHANISMS[k]}")
    print("\nPopulation table (booklet Ch. 4, the first table of any audit):")
    print(population_table(df).round(3).to_string(index=False))
    print("\nNote: both groups have IDENTICAL latent creditworthiness by construction.")
    print(f"  mean latent, group A: {df[df.group=='A'].latent.mean():+.3f}")
    print(f"  mean latent, group B: {df[df.group=='B'].latent.mean():+.3f}")
    print("Every disparity below was manufactured by the mechanisms above.")

    rule("STAGE 2 · IS THE PROTECTED ATTRIBUTE RECOVERABLE? (Ch. 5)")
    rec = proxy_recoverability(df)
    print(f"  predicting `group` from the permitted features alone:")
    print(f"    accuracy          {rec.attrs['accuracy']:.3f}")
    print(f"    majority baseline {rec.attrs['majority_baseline']:.3f}")
    print("\n  per-feature contribution to that recovery:")
    print(rec.round(4).to_string(index=False))
    print("\n  'Fairness through unawareness' has already failed, and we have not")
    print("  trained the actual model yet.")

    rule("STAGE 5-6 · THE MODEL AND THE THRESHOLD (Ch. 7)")
    model, train, test, scores = train_model(df, seed=1)
    y, a = test.repaid.to_numpy(), test.group.to_numpy()
    print(fm.summarise(y, scores, a, 0.5))

    ex.fig_distributions(y, scores, a, 0.5).savefig(out / "01_distributions.png", bbox_inches="tight")
    fig, sweep = ex.fig_threshold_sweep(y, scores, a)
    fig.savefig(out / "02_threshold_sweep.png", bbox_inches="tight")
    ex.fig_calibration(y, scores, a).savefig(out / "03_calibration.png", bbox_inches="tight")

    rule("STAGE 6 · THE IMPOSSIBILITY RESULT, BY EXPERIMENT (Ch. 9)")
    fig, table = ex.fig_impossibility(y, scores, a)
    fig.savefig(out / "04_impossibility.png", bbox_inches="tight")
    print(table.to_string(index=False))
    print("\n  Each row tunes per-group thresholds to zero ONE gap. Look along the")
    print("  row: the other gaps do not follow. No row zeroes all three, because")
    print("  no such configuration exists.")

    rule("EXPLANATION · LOCAL AND GLOBAL (Ch. 12)")
    import explain
    refused_idx = test[(scores < 0.4) & (test.group == "B")].index[0]
    x = test.loc[refused_idx, FEATURES]
    local = explain.lime_from_scratch(model, train[FEATURES], x)
    print("Local explanation for one refused group-B applicant (LIME, from scratch):")
    print(local.round(4).to_string())
    explain.fig_local_explanation(local).savefig(out / "05_local_explanation.png",
                                                 bbox_inches="tight")
    print("\nGlobal importance (permutation, AUC):")
    gi = explain.global_importance(model, test[FEATURES], test.repaid)
    print(gi.round(4).to_string(index=False))
    print("\n  An explanation tells you what the model USED. It does not tell you")
    print("  whether the model SHOULD have. That judgement is yours.")

    rule("MITIGATION · FOUR ATTEMPTS AND WHAT EACH COSTS (Ch. 15)")
    import mitigate
    tbl = mitigate.compare_all(n=8000, seed=1)
    print(tbl.round(4).to_string(index=False))

    rule("WHAT TO TAKE AWAY")
    print("""  1. The groups were identical in ability. Every gap you measured was
     manufactured upstream of the model.
  2. Removing the protected attribute achieved nothing: `group` stayed
     recoverable at near-perfect accuracy from the permitted features.
  3. No threshold satisfies every fairness criterion, and the impossibility
     table shows this is structural rather than a tuning failure.
  4. Model-side patches trade one gap for another. Fixing the LABEL fixed
     nearly all of them, at almost no accuracy cost, because that is where
     the harm was introduced.

  Now write the model card. Templates are in this folder.""")
    print(f"\nFigures written to {out.resolve()}")


if __name__ == "__main__":
    main()
