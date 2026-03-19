#!/usr/bin/env bash
set -euo pipefail

if conda env list | rg -q '^is3d-ml\s'; then
  conda env update -f environment.ml.yml --prune
else
  conda env create -f environment.ml.yml
fi

echo "Run: conda activate is3d-ml"
