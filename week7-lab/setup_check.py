"""
setup_check.py — does your machine have what Weeks 3 to 12 will need?

The Evolution of Machine Learning and AI · Week 1 Lab
South East Technological University

Run this first. It never crashes and it never installs anything: it looks,
reports, and tells you the exact command to fix whatever is missing.

    python setup_check.py

Checks: Python version, the scientific stack, PyTorch and a usable accelerator,
HuggingFace (library, cache, token), Ollama (daemon and models), git identity,
and whether you are in Colab.
"""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys

OK, WARN, BAD = "  OK  ", " NOTE ", " MISS "
results: list[tuple[str, str, str]] = []


def record(status, name, detail=""):
    results.append((status, name, detail))
    print(f"[{status}] {name:34s} {detail}")


def check_python():
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 10):
        record(OK, "Python", f"{ver} on {platform.system()} {platform.machine()}")
    else:
        record(BAD, "Python", f"{ver} — this module assumes 3.10 or newer")


def check_packages():
    core = ["numpy", "pandas", "sklearn", "matplotlib", "scipy"]
    week_later = ["torch", "transformers", "datasets", "shap", "lime", "ipywidgets"]
    for pkg in core + week_later:
        try:
            m = importlib.import_module(pkg)
            ver = getattr(m, "__version__", "?")
            record(OK, pkg, ver)
        except Exception:
            need = "required" if pkg in core else "needed from Week 3 onward"
            record(BAD if pkg in core else WARN, pkg,
                   f"not installed ({need}) — bash setup.sh --full")


def check_accelerator():
    try:
        import torch
    except Exception:
        record(WARN, "accelerator", "torch not installed; skipped")
        return
    if torch.cuda.is_available():
        record(OK, "accelerator", f"CUDA — {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        record(OK, "accelerator", "Apple MPS")
    else:
        record(WARN, "accelerator", "CPU only — fine for Weeks 1-5; use Colab later")


def check_huggingface():
    try:
        importlib.import_module("huggingface_hub")
    except Exception:
        record(WARN, "huggingface_hub", "not installed — pip install huggingface_hub")
        return
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cached = os.path.isdir(os.path.join(home, "hub"))
    record(OK if token else WARN, "HuggingFace token",
           "found in environment" if token else
           "none set — fine for open models; needed for gated ones (Week 7)")
    record(OK if cached else WARN, "HuggingFace cache",
           home if cached else f"{home} (empty; first download will populate it)")


def check_ollama():
    """Ollama is the Week 7 and Week 9 runtime: local models, no API key."""
    exe = shutil.which("ollama")
    if not exe:
        record(WARN, "Ollama", "not installed — see https://ollama.com (needed Week 7)")
        return
    try:
        out = subprocess.run([exe, "list"], capture_output=True, text=True, timeout=8)
        if out.returncode != 0:
            record(WARN, "Ollama", "installed, daemon not responding — run: ollama serve")
            return
        lines = [l for l in out.stdout.strip().splitlines()[1:] if l.strip()]
        if lines:
            names = ", ".join(l.split()[0] for l in lines[:4])
            record(OK, "Ollama", f"{len(lines)} model(s): {names}")
        else:
            record(WARN, "Ollama", "running, no models — try: ollama pull llama3.2:1b")
    except Exception as e:
        record(WARN, "Ollama", f"present but not queryable ({type(e).__name__})")


def check_git():
    exe = shutil.which("git")
    if not exe:
        record(BAD, "git", "not installed — you will need it for every submission")
        return
    try:
        name = subprocess.run([exe, "config", "--get", "user.name"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        email = subprocess.run([exe, "config", "--get", "user.email"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        if name and email:
            record(OK, "git identity", f"{name} <{email}>")
        else:
            record(WARN, "git identity", 'not set — git config --global user.name "You"')
    except Exception:
        record(WARN, "git identity", "could not read config")


def check_colab():
    in_colab = "google.colab" in sys.modules or os.path.isdir("/content")
    record(OK if in_colab else WARN, "environment",
           "Google Colab" if in_colab else "local machine (Colab also fine)")


def main():
    print("=" * 72)
    print("Week 1 Lab · environment check")
    print("=" * 72)
    check_python(); check_colab()
    print("-" * 72); check_packages()
    print("-" * 72); check_accelerator()
    print("-" * 72); check_huggingface(); check_ollama(); check_git()
    print("=" * 72)

    bad = [r for r in results if r[0] == BAD]
    warn = [r for r in results if r[0] == WARN]
    print(f"{len(results) - len(bad) - len(warn)} ready · {len(warn)} to sort out later "
          f"· {len(bad)} blocking")
    if bad:
        print("\nBlocking, fix before Week 3:")
        for _, name, detail in bad:
            print(f"  - {name}: {detail}")
    else:
        print("\nNothing blocking. You are ready for the timeline exercise.")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
