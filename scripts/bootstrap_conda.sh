#!/usr/bin/env bash
set -euo pipefail

if conda env list | rg -q '^is3d\s'; then
  conda env update -f environment.yml --prune
else
  conda env create -f environment.yml
fi

echo "Run: conda activate is3d"
