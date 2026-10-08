#!/usr/bin/env bash
# One-command setup for the Week 2 lab (macOS / Linux).
#   bash setup.sh
# Creates a virtual environment, installs the requirements, and runs the check.
#
# The core five packages are installed first and must succeed. shap, lime and
# ipywidgets are attempted afterwards and are allowed to fail: the lab has a
# from-scratch LIME and does not depend on them.
set -euo pipefail

PY=${PYTHON:-python3}

echo "==> Using $($PY --version)"
$PY -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"' \
  || { echo "Install Python 3.10 or newer, then re-run."; exit 1; }

if [ ! -d .venv ]; then
  echo "==> Creating .venv"
  $PY -m venv .venv
else
  echo "==> .venv already exists, reusing it"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "==> Upgrading pip"
python -m pip install --quiet --upgrade pip

echo "==> Installing core requirements (numpy, pandas, scikit-learn, matplotlib, scipy)"
python -m pip install --quiet -r requirements-core.txt

echo "==> Installing optional extras (shap, lime, ipywidgets)"
if python -m pip install --quiet -r requirements.txt; then
  echo "    ok"
else
  echo "    One or more optional packages failed to build on this machine."
  echo "    That is not blocking. explain.py has a from-scratch LIME, and the"
  echo "    lab runs end to end without shap, lime or ipywidgets."
fi

echo
echo "==> Environment check"
python setup_check.py || true

echo
echo "==> Smoke test: the whole lab, end to end (about 40 seconds)"
python run_lab.py > /dev/null && echo "    run_lab.py completed; figures are in ./figures"

echo
echo "Done. Activate this environment in future terminals with:"
echo "    source .venv/bin/activate"
