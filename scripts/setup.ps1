# Windows setup: venv, CPU/CUDA torch via pip, backend + frontend deps.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
# Default PyPI torch (NVIDIA CUDA build when available). For CPU-only:
#   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch==2.5.1 torchvision==0.20.1
pip install -r backend\requirements.txt -r backend\requirements-dev.txt
Set-Location frontend
npm install
Set-Location $Root
New-Item -ItemType Directory -Force -Path data\datasets, data\workspaces, data\uploads | Out-Null
Write-Host "YoloFlow setup complete. Activate with .\.venv\Scripts\Activate.ps1"
