#!/usr/bin/env bash
# Sets up the Nemesis dev environment on Linux (e.g. a Docker Agentic Platform sandbox):
# CybORG (CAGE Challenge 4) in a Python 3.10 venv. Linux counterpart of setup.ps1.
#
#   bash setup.sh            # CPU-only PyTorch (default; CybORG training is CPU-bound)
#   TORCH_CUDA=1 bash setup.sh   # default PyPI torch wheel, which bundles CUDA (~2 GB)
set -euo pipefail
cd "$(dirname "$0")"

git submodule update --init --recursive

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# CC4 pins numpy 1.26 / torch 2.2, which have no wheels for Python >= 3.13.
uv venv --allow-existing --python 3.10 .venv
PY=.venv/bin/python

if [ "${TORCH_CUDA:-0}" != "1" ]; then
    # Pre-install the CPU build; it satisfies CC4's torch==2.2.0 pin.
    uv pip install --python "$PY" torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
fi

uv pip install --python "$PY" -r external/cage-challenge-4/Requirements.txt

# CC4's setup.py declares no packages, so a default (strict) editable install maps nothing
# and `import CybORG` fails outside the repo. Compat mode puts the repo dir on sys.path.
uv pip install --python "$PY" -e external/cage-challenge-4 --config-settings editable_mode=compat

"$PY" scripts/smoke_test.py
