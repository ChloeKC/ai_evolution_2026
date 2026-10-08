"""
run_lab.py, the whole Week 4 lab, end to end.

    python run_lab.py              full run
    python run_lab.py --quick      smaller sweeps, for a laptop or a demo

Regenerates every figure and every table used in the lecture and the booklet,
and writes `results.json` so that nothing in the teaching materials is a number
somebody typed from memory.

The Evolution of Machine Learning and AI, Week 4 Lab
South East Technological University

This is a heavier lab than Week 3. Week 3 trained seven small classical models
and one tiny network; this week trains convolutional, recurrent and attention
models with every derivative written out in numpy, on a single core, so the
full run takes minutes rather than seconds. Use --quick if you are impatient or
demonstrating live. The PyTorch section is skipped with a message if torch is
not installed, and nothing else depends on it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78, flush=True)


def main():
    quick = "--quick" in sys.argv
    out = Path("figures")
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 210)
    pd.set_option("display.max_colwidth", 70)
    results = {"quick": quick}
    t0 = time.time()

    # -----------------------------------------------------------------
    rule("1 . CONVOLUTION: THE OPERATION, AND WHAT IT ASSUMES  (booklet Ch. 6-9)")
    import convolution as cv

    print("\nSix hand-designed kernels on one synthetic image")
    kt = cv.kernel_table()
    print(kt.to_string(index=False))

    print("\nA dense layer against a convolution layer, on the same input")
    pc = cv.parameter_comparison()
    print(pc.to_string(index=False))
    print("\n  The dense layer is not being treated unfairly. It is doing exactly")
    print("  what you asked: considering every pixel's relationship to every other.")
    print("  The problem is that you did not want that.")

    print("\nThe receptive field, growing")
    rf = cv.receptive_field_table()
    print(rf.to_string(index=False))

    print("\nTranslation equivariance, tested rather than asserted")
    eq = cv.equivariance_test()
    print(eq.to_string(index=False))

    cv.fig_feature_maps().savefig(out / "01_feature_maps.png", bbox_inches="tight", dpi=120)
    cv.fig_stride_padding().savefig(out / "02_stride_padding.png", bbox_inches="tight", dpi=120)
    cv.fig_receptive_field().savefig(out / "03_receptive_field.png", bbox_inches="tight", dpi=120)
    cv.fig_equivariance().savefig(out / "04_equivariance.png", bbox_inches="tight", dpi=120)
    cv.fig_learned_filters().savefig(out / "05_learned_filters.png", bbox_inches="tight", dpi=120)

    results["kernels"] = kt.to_dict("records")
    results["parameter_comparison"] = pc.to_dict("records")
    results["receptive_field"] = rf.to_dict("records")
    results["equivariance"] = eq.to_dict("records")

    # -----------------------------------------------------------------
    rule("2 . THE HAND-WRITTEN CONVOLUTION BACKWARD PASS  (booklet Ch. 5)")
    import cnn_numpy as cn

    rel, loss = cn.gradient_check()
    print(f"\nConv2D gradient check: max relative error {rel:.3e}")
    print("Below about 1e-5 means the derivative written out by hand is correct.")
    print("Week 3, Chapter 14: a wrong derivative fails silently, so you check it once.")
    print(f"\nCNN (global pooling) parameters: {cn.make_cnn().n_params:,}")
    print(f"CNN (flatten head) parameters:   {cn.make_cnn_flatten().n_params:,}")
    print(f"MLP parameters:                  {cn.make_mlp().n_params:,}")
    results["gradient_check_conv"] = rel
    results["params"] = {
        "cnn_globalpool": cn.make_cnn().n_params,
        "cnn_flatten": cn.make_cnn_flatten().n_params,
        "mlp": cn.make_mlp().n_params,
    }

    # -----------------------------------------------------------------
    rule("3 . STRUCTURE IS AN ASSUMPTION, MEASURED FOUR WAYS  (booklet Ch. 2, 3, 11)")
    import inductive_bias as ib

    print("\nEXPERIMENT 1: data efficiency")
    eff = ib.experiment_data_efficiency(quick=quick)
    print(eff.to_string(index=False))

    print("\n  ...and the MLP given far more data and four times the capacity")
    scaling = ib.experiment_mlp_scaling(quick=quick)
    print(scaling.to_string(index=False))

    cnn_row = eff.iloc[min(1, len(eff) - 1)]
    best = scaling.loc[scaling["test acc"].idxmax()]
    print(f"\n  The CNN scored {cnn_row['CNN test']:.3f} from {int(cnn_row['train images'])} images "
          f"with {cn.make_cnn().n_params:,} parameters.")
    print(f"  The MLP, given {int(best['MLP train images']):,} images and "
          f"{int(best['parameters']):,} parameters, scored {best['test acc']:.3f}.")
    ratio = int(best["MLP train images"]) // int(cnn_row["train images"])
    print(f"  That is roughly {ratio} times the data, for a model that is still behind.")

    print("\nEXPERIMENT 2: positions never seen during training")
    unseen = ib.experiment_unseen_positions()
    print(unseen.to_string(index=False))
    print("\n  Rows 1 and 2 have IDENTICAL convolutional layers. They differ only in")
    print("  the last two lines of the architecture. An assumption is worth only what")
    print("  your architecture does with it all the way to the output.")

    print("\nEXPERIMENT 3: one fixed permutation of the pixels")
    shuf, perm = ib.experiment_shuffle()
    print(shuf.to_string(index=False))
    print("\n  No information was destroyed: the permutation is reversible and the label")
    print("  is still perfectly determined. Only the MEANING of adjacency is gone, and")
    print("  adjacency is the only thing the convolution ever assumed.")

    print("\nEXPERIMENT 4: where the assumption does NOT pay")
    nopay = ib.experiment_no_payoff()
    print(nopay.to_string(index=False))
    print("\n  Report this one honestly. Pre-centred, pre-cropped 8 by 8 digits have had")
    print("  the translation removed already, in the preprocessing step. You do not get")
    print("  paid twice for solving the same problem once.")

    ib.fig_data_efficiency(eff, scaling, cnn_ref=float(cnn_row["CNN test"])).savefig(
        out / "06_data_efficiency.png", bbox_inches="tight", dpi=120)
    ib.fig_shuffle(perm).savefig(out / "07_pixel_shuffle.png", bbox_inches="tight", dpi=120)
    ib.fig_summary(eff, unseen, shuf).savefig(
        out / "08_three_experiments.png", bbox_inches="tight", dpi=120)

    results["data_efficiency"] = eff.to_dict("records")
    results["mlp_scaling"] = scaling.to_dict("records")
    results["unseen_positions"] = unseen.to_dict("records")
    results["pixel_shuffle"] = shuf.to_dict("records")
    results["no_payoff"] = nopay.to_dict("records")

    # -----------------------------------------------------------------
    rule("4 . SEQUENCES: RECURRENCE AND ITS PRICE  (booklet Ch. 12-16)")
    import sequences as sq

    print("\nEXPERIMENT 1: the gradient through time, measured")
    vt, vrows = sq.gradient_through_time()
    print(vt.to_string(index=False))
    print("\n  A product of T copies of ONE matrix. In a deep feedforward network the")
    print("  layers have different matrices, so the decay is partly luck. Here it is")
    print("  systematic. Depth in time is worse than depth in layers.")

    print("\nEXPERIMENT 2: remember the first symbol across T steps (chance is 0.250)")
    mem, curves = sq.experiment_memory(quick=quick)
    print(mem.to_string(index=False))

    print("\nStructural comparison: how far is position 1 from position n?")
    pl = sq.path_length_table()
    print(pl.to_string(index=False))

    sq.fig_vanishing(vrows).savefig(out / "09_vanishing_time.png", bbox_inches="tight", dpi=120)
    sq.fig_memory(mem, curves).savefig(out / "10_memory_task.png", bbox_inches="tight", dpi=120)

    results["gradient_through_time"] = vt.to_dict("records")
    results["memory_task"] = mem.to_dict("records")
    results["path_lengths"] = pl.to_dict("records")

    # -----------------------------------------------------------------
    rule("5 . ATTENTION: RELATIONS AS A FIRST-CLASS OBJECT  (booklet Ch. 17-21)")
    import attention as at

    E, scores, weights, _ = at.worked_example()
    print("\nEXPERIMENT 1: six tokens. Raw scores, Q K^T / sqrt(d):")
    print(scores.to_string())
    print("\n  After softmax, every row adds to 1:")
    print(weights.to_string())
    row = weights.loc["are"]
    dist = len(at.TOKENS) - 1 - at.TOKENS.index("keys")
    print(f"\n  \"are\" gives {row['keys']:.3f} to \"keys\", {dist} positions away, and only")
    print(f"  {row['left']:.3f} to \"left\", which is adjacent. Content decided it, not distance.")

    print("\nEXPERIMENT 2: does attention know what order the tokens arrived in?")
    perm_t = at.permutation_test()
    print(perm_t.to_string(index=False))
    print("\n  That first number is zero to machine precision, and it is the entire")
    print("  reason positional encoding exists.")

    gc = at.gradient_check()
    print(f"\nGradient check on the hand-written attention backward pass: {gc:.3e}")

    print("\nEXPERIMENT 3: content lookup, attention against an LSTM (chance is 0.250)")
    look = at.experiment_lookup(quick=quick)
    print(look.to_string(index=False))
    print("\n  Read the attention column down the page. The task gets longer and the")
    print("  score does not fall, because attention does not care how far away the")
    print("  answer is. The LSTM has to carry every pair in a fixed-size state.")

    print("\nEXPERIMENT 4: the bill")
    ct = at.cost_table()
    print(ct.to_string(index=False))

    at.fig_attention_matrix(weights).savefig(out / "11_attention_matrix.png", bbox_inches="tight", dpi=120)
    at.fig_positional_encoding().savefig(out / "12_positional_encoding.png", bbox_inches="tight", dpi=120)
    at.fig_multi_head().savefig(out / "13_multi_head.png", bbox_inches="tight", dpi=120)
    at.fig_cost().savefig(out / "14_cost_and_path.png", bbox_inches="tight", dpi=120)
    at.fig_learned_lookup().savefig(out / "15_learned_lookup.png", bbox_inches="tight", dpi=120)

    results["attention_weights"] = weights.round(3).to_dict()
    results["permutation_test"] = perm_t.to_dict("records")
    results["gradient_check_attention"] = gc
    results["lookup_task"] = look.to_dict("records")
    results["cost_table"] = ct.to_dict("records")

    # -----------------------------------------------------------------
    rule("6 . THE SAME THREE FAMILIES IN PYTORCH  (booklet Ch. 22)")
    try:
        import torch_models
        torch_models.main()
        results["torch"] = "ran"
    except ImportError as e:
        print(f"\nPyTorch not installed, skipping ({e}).")
        print("Everything above runs without it. To install:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
        results["torch"] = "skipped"

    # -----------------------------------------------------------------
    rule("WHAT TO TAKE AWAY")
    print("""  1. A convolution is two assumptions written as arithmetic: meaning is
     LOCAL, and a feature worth detecting anywhere is worth detecting
     everywhere. Everything else about a CNN follows from those two sentences.
  2. Those assumptions are worth data. The CNN matched a score the MLP could
     not reach with orders of magnitude more examples and far more parameters.
  3. The pixel shuffle is the proof. Destroy adjacency and the CNN's entire
     advantage disappears, while the model that assumed nothing loses nothing.
  4. An assumption must be carried all the way to the output. The same
     convolutional trunk with a position-reading head is back at chance on
     positions it never saw.
  5. Recurrence shares weights across TIME instead of space, which is the same
     trick rotated. The price is a gradient that is a power of one matrix.
  6. Attention makes the relation between two positions the object being
     computed, so distance stops mattering. It pays quadratic compute, and it
     does not know what order the tokens came in until you tell it.
  7. Weaker assumptions need more data. That is the bargain of Week 4 and the
     premise of Week 5, which asks what happens when you can put real physical
     law into the structure instead.""")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nFigures written to {out.resolve()}")
    print(f"Numbers written to {Path('results.json').resolve()}")
    print(f"Total run time: {time.time() - t0:.0f} seconds"
          f"{' (quick mode)' if quick else ''}")


if __name__ == "__main__":
    main()
