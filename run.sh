#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

if [ ! -d ".venv" ]; then
    $PY -m venv .venv
fi

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    source .venv/Scripts/activate
fi

python -m pip install --upgrade pip -q
python -m pip install scikit-learn -q

python bcw_mlp_logreg.py