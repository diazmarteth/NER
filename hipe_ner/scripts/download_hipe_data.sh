#!/usr/bin/env bash
# Fetches the HIPE-2022 shared-task data (~200MB, includes v1.0/v2.0/v2.1;
# we only use v2.1). Not committed to git — see .gitignore.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d data/raw/.git ]; then
    echo "data/raw already present, skipping clone (git pull manually to update)"
else
    git clone --depth 1 https://github.com/hipe-eval/HIPE-2022-data.git data/raw
fi
echo "done. TSVs live under data/raw/data/v2.1/<dataset>/<lang>/"
