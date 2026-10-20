# Sets up the Nemesis dev environment: CybORG (CAGE Challenge 4) in a Python 3.10 venv.
# Requires git and uv (https://docs.astral.sh/uv/). Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

git submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw "git submodule update failed" }

# CC4 pins numpy 1.26 / torch 2.2, which have no wheels for Python >= 3.13.
uv venv --python 3.10 .venv
if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }

uv pip install --python .venv\Scripts\python.exe -r external\cage-challenge-4\Requirements.txt
if ($LASTEXITCODE -ne 0) { throw "installing CC4 requirements failed" }

# CC4's setup.py declares no packages, so a default (strict) editable install maps nothing
# and `import CybORG` fails outside the repo. Compat mode puts the repo dir on sys.path.
uv pip install --python .venv\Scripts\python.exe -e external\cage-challenge-4 --config-settings editable_mode=compat
if ($LASTEXITCODE -ne 0) { throw "installing CybORG failed" }

.venv\Scripts\python.exe scripts\smoke_test.py
