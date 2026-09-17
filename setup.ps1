# One-command setup for the Week 1 lab (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File setup.ps1
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

param([switch]$Full)
$req = if ($Full) { "requirements-full.txt" } else { "requirements.txt" }
Write-Host "==> Installing $req"
if (-not $Full) { Write-Host "    For the Week 3+ stack, re-run later with:  .\setup.ps1 -Full" }
python -m pip install --quiet -r $req

Write-Host ""
Write-Host "==> Environment check"
python setup_check.py

Write-Host ""
Write-Host "Done. Activate this environment in future terminals with:"
Write-Host "    .\.venv\Scripts\activate"
