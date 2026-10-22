"""
run_lab.py, the whole Week 5 lab, end to end.

    python run_lab.py              full run
    python run_lab.py --quick      smaller sweeps, for a laptop or a demo

Regenerates every figure and table used in the lecture and the booklet, and
writes `results.json`, so that nothing in the teaching materials is a number
somebody typed from memory.

The Evolution of Machine Learning and AI, Week 5 Lab
South East Technological University

This lab is heavier than Week 4's, for a reason worth knowing. Every physics
residual requires derivatives of the network with respect to its INPUT, computed
inside the training loop, and then differentiated again with respect to the
weights. A PINN epoch therefore costs several times what an ordinary epoch costs,
and there is no way around it: it is the price of the capability. On one core the
full run takes several minutes. Use --quick if you are impatient or demonstrating
live.

Nothing here imports a deep learning library. The autodiff engine in
autodiff.py is about 250 lines and you have read it.
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


SECTIONS = ["engine", "pinn", "heat", "symmetry", "ode", "operator", "diffusion"]


def main():
    quick = "--quick" in sys.argv
    # --only lets you run part of the lab and come back to the rest later, which
    # matters on a slow machine: the full run takes several minutes because every
    # physics residual needs derivatives of the network inside the training loop.
    #     python run_lab.py --only pinn,heat
    wanted = SECTIONS
    for a in sys.argv:
        if a.startswith("--only"):
            wanted = a.split("=", 1)[1].split(",") if "=" in a else \
                sys.argv[sys.argv.index(a) + 1].split(",")
    run = lambda name: name in wanted

    out = Path("figures")
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 210)
    pd.set_option("display.max_colwidth", 60)
    # Results merge with anything already on disk, so partial runs accumulate.
    results = {}
    if Path("results.json").exists():
        try:
            results = json.load(open("results.json"))
        except Exception:
            results = {}
    results["quick"] = quick
    t0 = time.time()

    # -----------------------------------------------------------------
    if run("engine"):
      rule("0 . THE ENGINE: DIFFERENTIATING A DERIVATIVE  (booklet Ch. 7)")
      import autodiff as ad
      import mlp as M

      print("\nDerivatives of known functions, to third order:")
      worst = ad.check_engine()
      print(f"\n  worst relative error: {worst:.2e}")
      print("\n  Exact to machine precision, because the engine differentiates")
      print("  symbolically. Compare that against a finite difference:")
      r1, r2 = M.check_network_derivative()
      print("\n  The disagreement is the finite difference's error, not the engine's.")
      print("  Chapter 8 explains why that distinction decides whether a PINN works.")
      results["engine_worst_error"] = worst
      results["finite_difference_gap"] = {"first": r1, "second": r2}

    # -----------------------------------------------------------------
    if run("pinn"):
      rule("1 . A PINN FOR A DAMPED OSCILLATOR  (booklet Ch. 8-12)")
      import pinn as P

      system = P.DampedOscillator()
      print(f"\nThe system: {system.describe()}")
      print("\nBoth models are the SAME network, the same size, the same optimiser.")
      print("The only thing that differs is the loss function.")

      print("\nEXPERIMENT 1: data efficiency")
      eff, trained = P.experiment_data_efficiency(quick=quick)
      print(eff.to_string(index=False))
      print("\n  Read the physics-informed row's observation count. It is zero.")
      print("  Not few: none. It was given the equation and the initial condition")
      print("  and it reconstructed the trajectory to within about a thousandth.")

      print("\nEXPERIMENT 2: beyond the data")
      ext, models, obs = P.experiment_extrapolation(quick=quick)
      print(ext.to_string(index=False))
      print("\n  This is the reliability argument. A model that is excellent inside")
      print("  its training range and arbitrary outside it is dangerous precisely")
      print("  because the failure is silent.")

      print("\nEXPERIMENT 3: measurement noise")
      noise = P.experiment_noise(quick=quick)
      print(noise.to_string(index=False))
      print("\n  A physics residual is a regulariser that penalises IMPOSSIBILITY")
      print("  rather than complexity, and nobody had to choose its strength.")

      print("\nEXPERIMENT 4: how the initial condition is imposed")
      cs = P.experiment_constraint_style(quick=quick)
      print(cs.to_string(index=False))
      print("\n  Note the u(0) column: every version satisfies the condition. The bad")
      print("  runs are not violating it, they are satisfying it while neglecting the")
      print("  equation, because their loss said it mattered five hundred times more.")

      print("\nEXPERIMENT 5: WHEN THE PHYSICS IS WRONG")
      wrong, wmodels = P.experiment_wrong_physics(quick=quick)
      print(wrong.to_string(index=False))
      print("\n  Report this one. A wrong prior is worse than no prior, and the last")
      print("  column is why: the physics loss stays small and healthy-looking the")
      print("  whole way, so the failure does not announce itself.")

      print("\nEXPERIMENT 6: conservation of energy, which nobody put in a loss")
      cons, curves, meta = P.experiment_conservation(quick=quick)
      print(cons.to_string(index=False))
      print("\n  Energy is a CONSEQUENCE of the equation. A model that satisfies the")
      print("  equation inherits it; a model that fits points does not.")

      P.fig_solution(trained).savefig(out / "01_pinn_solution.png", bbox_inches="tight", dpi=120)
      P.fig_data_efficiency(eff).savefig(out / "02_data_efficiency.png", bbox_inches="tight", dpi=120)
      P.fig_extrapolation(models, obs).savefig(out / "03_extrapolation.png", bbox_inches="tight", dpi=120)
      P.fig_wrong_physics(wmodels).savefig(out / "04_wrong_physics.png", bbox_inches="tight", dpi=120)
      P.fig_conservation(curves, meta).savefig(out / "05_conservation.png", bbox_inches="tight", dpi=120)

      results["pinn"] = {
          "data_efficiency": eff.to_dict("records"),
          "extrapolation": ext.to_dict("records"),
          "noise": noise.to_dict("records"),
          "constraint_style": cs.to_dict("records"),
          "wrong_physics": wrong.to_dict("records"),
          "conservation": cons.to_dict("records"),
      }

    # -----------------------------------------------------------------
    if run("heat"):
      rule("2 . THE SAME IDEA ON A PDE: THE HEAT EQUATION  (booklet Ch. 13)")
      import heat as H

      hs = H.HeatEquation()
      print(f"\nThe system: {hs.describe()}")
      print("\nThe hard constraint u(x,t) = sin(pi x)[1 + t N(x,t)] satisfies the")
      print("initial condition and BOTH boundaries exactly, for any weights, so the")
      print("loss has exactly one term and there is nothing to balance.")
      hdf, hp = H.experiment_heat(quick=quick)
      print()
      print(hdf.to_string(index=False))

      H.fig_heat(hp).savefig(out / "06_heat_field.png", bbox_inches="tight", dpi=120)
      H.fig_heat_slices(hp).savefig(out / "07_heat_slices.png", bbox_inches="tight", dpi=120)
      results["heat"] = hdf.to_dict("records")

    # -----------------------------------------------------------------
    if run("symmetry"):
      rule("3 . SYMMETRY: THE ONLY PRIOR THAT IS EXACTLY TRUE  (booklet Ch. 14-15)")
      import symmetry as S

      print("\nEXPERIMENT 1: is the invariance exact, or only learned?")
      inv = S.experiment_invariance()
      print(inv.to_string(index=False))
      print("\n  Compare Week 4's equivariance test on convolution, which also")
      print("  returned exactly zero. Same argument, different symmetry group.")

      print("\nEXPERIMENT 2: data efficiency")
      seff = S.experiment_symmetry_efficiency(quick=quick)
      print(seff.to_string(index=False))
      print("\n  Note that the coordinate model PLATEAUS. More data does not close")
      print("  the gap, because it is spending capacity representing a fact the")
      print("  other model was handed for nothing.")

      print("\nEXPERIMENT 3: orientations never seen in training")
      sun = S.experiment_unseen_orientations(quick=quick)
      print(sun.to_string(index=False))

      print("\nEXPERIMENT 4: when the symmetry does NOT hold")
      swr = S.experiment_wrong_symmetry(quick=quick)
      print(swr.to_string(index=False))
      print("\n  Add a gravity term and the energy genuinely does depend on")
      print("  orientation. Now the invariant model cannot represent the truth at")
      print("  any dataset size. A constraint is a promise, and a false promise is")
      print("  not a handicap, it is a wall.")

      S.fig_symmetry_efficiency(seff).savefig(out / "08_symmetry_efficiency.png", bbox_inches="tight", dpi=120)
      S.fig_rotation_sweep().savefig(out / "09_rotation_sweep.png", bbox_inches="tight", dpi=120)
      results["symmetry"] = {
          "invariance": inv.to_dict("records"),
          "efficiency": seff.to_dict("records"),
          "unseen_orientations": sun.to_dict("records"),
          "wrong_symmetry": swr.to_dict("records"),
      }

    # -----------------------------------------------------------------
    if run("ode"):
      rule("4 . NEURAL ODEs: LEARNING THE LAW, NOT THE ANSWER  (booklet Ch. 16-17)")
      import neural_ode as NO

      print("\nEXPERIMENT 1: rolling out three times past the training window")
      roll = NO.experiment_rollout(quick=quick)
      print(roll.to_string(index=False))

      print("\nEXPERIMENT 2: initial conditions outside the training box")
      nic = NO.experiment_new_initial_conditions(quick=quick)
      print(nic.to_string(index=False))
      print("\n  All three degrade outside the box, and structure buys a smaller")
      print("  error rather than immunity. Report that honestly.")

      NO.fig_vector_field(epochs=350 if quick else 600).savefig(
          out / "10_vector_field.png", bbox_inches="tight", dpi=120)
      NO.fig_rollout(epochs=350 if quick else 600).savefig(
          out / "11_rollout.png", bbox_inches="tight", dpi=120)
      results["neural_ode"] = {"rollout": roll.to_dict("records"),
                               "new_initial_conditions": nic.to_dict("records")}

    # -----------------------------------------------------------------
    if run("operator"):
      rule("5 . NEURAL OPERATORS: SOLVING A WHOLE FAMILY AT ONCE  (booklet Ch. 18-19)")
      import operators as OP

      odf, onet, unseen, speedup = OP.experiment_operator(quick=quick)
      print()
      print(odf.to_string(index=False))
      print(f"\n  Once trained, the operator answers a new problem about "
            f"{speedup:,.0f} times faster, and is less accurate on any single one.")
      print("  For a design loop over thousands of parameter values, that trade is")
      print("  usually the difference between feasible and impossible.")

      OP.fig_operator(onet, unseen).savefig(out / "12_operator.png", bbox_inches="tight", dpi=120)
      OP.fig_basis(onet).savefig(out / "13_operator_basis.png", bbox_inches="tight", dpi=120)
      results["operator"] = {"table": odf.to_dict("records"), "speedup": float(speedup)}

    # -----------------------------------------------------------------
    if run("diffusion"):
      rule("6 . DIFFUSION AS NON-EQUILIBRIUM THERMODYNAMICS  (booklet Ch. 20-21)")
      import diffusion as D

      print("\nThe forward process reaching equilibrium. Nothing here is learned:")
      eq = D.experiment_equilibrium()
      print(eq.to_string(index=False))

      print("\nSample quality against the number of reverse steps:")
      ddf, dmodels, ddata = D.experiment_diffusion(quick=quick)
      print(ddf.to_string(index=False))
      print("\n  Read the last row first: two independent samples of the real data")
      print("  differ by that much, so it is the floor and nothing can score below it.")

      D.fig_forward().savefig(out / "14_diffusion_forward.png", bbox_inches="tight", dpi=120)
      key = sorted(dmodels.keys())[-1]
      m, sched = dmodels[key]
      D.fig_reverse(m, sched, ddata).savefig(out / "15_diffusion_reverse.png", bbox_inches="tight", dpi=120)
      D.fig_score(m, sched).savefig(out / "16_diffusion_score.png", bbox_inches="tight", dpi=120)
      results["diffusion"] = {"equilibrium": eq.to_dict("records"),
                              "samples": ddf.to_dict("records")}

    # -----------------------------------------------------------------
    rule("WHAT TO TAKE AWAY")
    print("""  1. A physical law is a stronger prior than any statistical regularity,
     because it is exactly true rather than approximately true. The PINN
     solved the oscillator from ZERO observations, better than a data-only
     model given sixty-four.
  2. The gain is largest where data is scarcest, which is where science and
     engineering actually live. Nobody has three hundred million labelled
     turbine failures.
  3. Reliability matters more than accuracy. The physics-informed model's error
     did not grow outside the data; the data-only model's grew by a factor of
     nearly thirty, silently.
  4. You get consequences you did not ask for. Energy was in nobody's loss and
     was conserved to a fifth of a percent anyway.
  5. Symmetry is the cleanest case: exactly true, exactly enforced, exactly
     zero violation. This is Week 4's equivariance argument with a group that
     comes from physics rather than from convenience.
  6. A WRONG law is worse than no law. It is confidently wrong, and the loss
     you are monitoring stays small while it happens.
  7. Structure lets you learn a rule rather than an answer: a vector field
     instead of a trajectory, an operator instead of a solution, a reversal of
     a known process instead of a leap from noise.""")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nFigures written to {out.resolve()}")
    print(f"Numbers written to {Path('results.json').resolve()}")
    print(f"Total run time: {time.time() - t0:.0f} seconds"
          f"{' (quick mode)' if quick else ''}")


if __name__ == "__main__":
    main()
