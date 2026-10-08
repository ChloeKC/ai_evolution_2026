"""
run_lab.py, the whole Week 3 lab, end to end.

    python run_lab.py

Regenerates every figure and table used in the lecture. The PyTorch section is
skipped with a message if torch is not installed; nothing else depends on it.
"""
from pathlib import Path
import pandas as pd


def rule(t):
    print("\n" + "=" * 76); print(t); print("=" * 76)


def main():
    out = Path("figures"); out.mkdir(exist_ok=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)

    rule("1 · SEVEN HYPOTHESIS CLASSES, ONE DATASET  (booklet Ch. 3-6)")
    import classical
    print(classical.compare().to_string(index=False))
    classical.fig_boundaries().savefig(out / "04_boundaries.png", bbox_inches="tight")
    fig, ft = classical.fig_fit(); fig.savefig(out / "05_fit.png", bbox_inches="tight")
    print("\nCapacity, on a curve you can see:")
    print(ft.to_string(index=False))
    print("\n  Read the GAP column above, not the test column. A large gap is the")
    print("  signature of a model that memorised instead of generalising.")

    rule("2 · GRADIENT DESCENT  (booklet Ch. 8-10)")
    import gradient_descent as gd
    fig, tbl = gd.lr_sweep(); fig.savefig(out / "01_lr_sweep.png", bbox_inches="tight")
    print(tbl.to_string(index=False))
    gd.landscape_2d().savefig(out / "02_landscape.png", bbox_inches="tight")
    fig, lm = gd.local_minima_demo(); fig.savefig(out / "03_local_minima.png", bbox_inches="tight")
    print("\nThree starting points, one algorithm:")
    print(lm.to_string(index=False))
    print("\nMomentum is memory:")
    print(gd.compare_optimisers().to_string(index=False))

    rule("3 · BACKPROPAGATION FROM SCRATCH  (booklet Ch. 7, 11)")
    import subprocess, sys
    subprocess.run([sys.executable, "scratch_net.py"])

    rule("4 · THE SAME NETWORK IN PYTORCH  (booklet Ch. 13)")
    subprocess.run([sys.executable, "torch_intro.py"])

    rule("WHAT TO TAKE AWAY")
    print("""  1. Every model in section 1 draws a different SHAPE of boundary, because
     each encodes a different assumption. That is Week 2's inductive bias and
     Week 4's whole thesis, arriving early and in pictures.
  2. The learning rate is not a detail. Two orders of magnitude separate
     'crawling' from 'diverging', and the good region between them is narrow.
  3. Backpropagation is the chain rule applied systematically. You wrote it.
     `loss.backward()` replaced eight of your lines, not the idea behind them.
  4. A hidden layer buys you a RE-REPRESENTATION, not just more parameters.""")
    print(f"\nFigures written to {out.resolve()}")


if __name__ == "__main__":
    main()
