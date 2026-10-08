# One-command setup for the Week 2 lab (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# The core five packages are installed first and must succeed. shap, lime and
# ipywidgets are attempted afterwards and are allowed to fail: the lab has a
# from-scratch LIME and does not depend on them.
$ErrorActionPreference = "Stop"

Write-Host "==> Using $(python --version)"

if (-not (Test-Path .venv)) {
    Write-Host "==> Creating .venv"
    python -m venv .venv
} else {
    Write-Host "==> .venv already exists, reusing it"
}

& .\.venv\Scripts\Activate.ps1
Write-Host "==> Upgrading pip"
python -m pip install --quiet --upgrade pip

Write-Host "==> Installing core requirements (numpy, pandas, scikit-learn, matplotlib, scipy)"
python -m pip install --quiet -r requirements-core.txt

Write-Host "==> Installing optional extras (shap, lime, ipywidgets)"
$ErrorActionPreference = "Continue"
python -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "    One or more optional packages failed to build on this machine."
    Write-Host "    That is not blocking. explain.py has a from-scratch LIME, and the"
    Write-Host "    lab runs end to end without shap, lime or ipywidgets."
} else {
    Write-Host "    ok"
}
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==> Environment check"
python setup_check.py

Write-Host ""
Write-Host "==> Smoke test: the whole lab, end to end (about 40 seconds)"
python run_lab.py | Out-Null
Write-Host "    run_lab.py completed; figures are in .\figures"

Write-Host ""
Write-Host "Done. Activate this environment in future terminals with:"
Write-Host "    .\.venv\Scripts\activate"
