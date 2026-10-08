"""
setup_check.py, can this machine run the Week 2 lab?

The Evolution of Machine Learning and AI · Week 2 Lab
South East Technological University

Run this first. It never crashes and it never installs anything: it looks,
reports, and tells you the exact command to fix whatever is missing.

    python setup_check.py

Checks: Python version, the five core packages, the three optional
explainability packages, that the lab's own modules import, that the dataset
generator and the metrics actually produce numbers, git identity, and whether
you are in Colab.

Week 1's check was about the whole semester's stack. This one is narrower on
purpose: Week 2 needs nothing heavy, no GPU and no downloads. If this reports
0 blocking, the lab will run.
"""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

OK, WARN, BAD = "  OK  ", " NOTE ", " MISS "
results: list[tuple[str, str, str]] = []

HERE = Path(__file__).resolve().parent


def record(status, name, detail=""):
    results.append((status, name, detail))
    print(f"[{status}] {name:34s} {detail}")


def check_python():
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 10):
        record(OK, "Python", f"{ver} on {platform.system()} {platform.machine()}")
    else:
        record(BAD, "Python", f"{ver}, this module assumes 3.10 or newer")


def check_colab():
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        record(OK, "Google Colab", "running in Colab, nothing to install")
    else:
        record(OK, "Environment", "local machine")


def check_packages():
    core = ["numpy", "pandas", "sklearn", "matplotlib", "scipy"]
    optional = ["shap", "lime", "ipywidgets"]

    for pkg in core:
        try:
            m = importlib.import_module(pkg)
            record(OK, pkg, getattr(m, "__version__", "?"))
        except Exception:
            record(BAD, pkg, "not installed, pip install -r requirements-core.txt")

    why = {
        "shap": "explain.py falls back to its own LIME",
        "lime": "explain.py has a from-scratch LIME in thirty lines",
        "ipywidgets": "the notebook falls back to a plain threshold loop",
    }
    for pkg in optional:
        try:
            m = importlib.import_module(pkg)
            record(OK, pkg, getattr(m, "__version__", "?"))
        except Exception:
            record(WARN, pkg, f"not installed (optional), {why[pkg]}")


def check_lab_modules():
    """The lab's own files, imported from this folder."""
    sys.path.insert(0, str(HERE))
    for mod in ["datasets", "fairness", "explorer", "explain", "mitigate"]:
        if not (HERE / f"{mod}.py").exists():
            record(BAD, f"{mod}.py", "missing from this folder, re-download the lab")
            continue
        try:
            importlib.import_module(mod)
            record(OK, f"{mod}.py", "imports cleanly")
        except Exception as e:                                  # noqa: BLE001
            record(BAD, f"{mod}.py", f"failed to import: {type(e).__name__}: {e}")


def check_it_actually_runs():
    """Generate a small dataset, train, and compute one metric. End to end."""
    try:
        sys.path.insert(0, str(HERE))
        from datasets import make_lending_data, train_model      # noqa: PLC0415
        import fairness as fm                                    # noqa: PLC0415

        df = make_lending_data(n=1200, seed=0)
        _, _, test, scores = train_model(df, seed=0)
        y, a = test.repaid.to_numpy(), test.group.to_numpy()
        gap = fm.demographic_parity_difference(y, scores, a, 0.5)
        record(OK, "end-to-end smoke test",
               f"trained on {len(df)} rows; selection-rate gap {gap:.3f}")
    except Exception as e:                                       # noqa: BLE001
        record(BAD, "end-to-end smoke test",
               f"{type(e).__name__}: {e}, run `python run_lab.py` to see it in full")


def check_git():
    if not shutil.which("git"):
        record(WARN, "git", "not installed, needed for the Week 12 project, not today")
        return
    try:
        name = subprocess.run(["git", "config", "--get", "user.name"],
                              capture_output=True, text=True, timeout=15).stdout.strip()
        email = subprocess.run(["git", "config", "--get", "user.email"],
                               capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:                                            # noqa: BLE001
        name = email = ""
    if name and email:
        record(OK, "git identity", f"{name} <{email}>")
    else:
        record(WARN, "git identity", 'set it: git config --global user.name "Your Name"')


def check_writable():
    try:
        out = HERE / "figures"
        out.mkdir(exist_ok=True)
        probe = out / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        record(OK, "figures/ writable", str(out))
    except Exception as e:                                       # noqa: BLE001
        record(BAD, "figures/ writable", f"cannot write here: {e}")


def main():
    print("=" * 72)
    print("Week 2 Lab · environment check")
    print("=" * 72)
    check_python(); check_colab()
    print("-" * 72); check_packages()
    print("-" * 72); check_lab_modules(); check_writable()
    print("-" * 72); check_it_actually_runs(); check_git()
    print("=" * 72)

    bad = [r for r in results if r[0] == BAD]
    warn = [r for r in results if r[0] == WARN]
    print(f"{len(results) - len(bad) - len(warn)} ready · {len(warn)} to sort out later "
          f"· {len(bad)} blocking")
    if bad:
        print("\nBlocking, fix before the lab:")
        for _, name, detail in bad:
            print(f"  - {name}: {detail}")
    else:
        print("\nNothing blocking. Run `python run_lab.py` and start the audit.")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
