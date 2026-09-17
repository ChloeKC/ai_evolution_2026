#!/usr/bin/env bash
# One-command setup for the Week 1 lab (macOS / Linux).
#   bash setup.sh
# Creates a virtual environment, installs the requirements, and runs the check.
set -euo pipefail

PY=${PYTHON:-python3}

echo "==> Using $($PY --version)"
$PY -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"' \
  || { echo "Install Python 3.11 or newer, then re-run."; exit 1; }

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

REQ=requirements.txt
if [ "${1:-}" = "--full" ]; then
  REQ=requirements-full.txt
  echo "==> Installing FULL requirements, including PyTorch (~2 GB, several minutes)"
else
  echo "==> Installing core requirements (small and fast)"
  echo "    For the Week 3+ stack, re-run later with:  bash setup.sh --full"
fi
python -m pip install --quiet -r "$REQ"

echo
echo "==> Environment check"
python setup_check.py || true

echo
echo "Done. Activate this environment in future terminals with:"
echo "    source .venv/bin/activate"
