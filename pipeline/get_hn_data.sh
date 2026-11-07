#!/usr/bin/env bash
# Run this in your own Mac Terminal (not inside Claude) to fetch the HN dump (~12.4 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install -U --user "huggingface_hub[cli]" >/dev/null
CLI=$(command -v hf || echo "$HOME/Library/Python/$(python3 -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')/bin/hf")
"$CLI" download open-index/hacker-news --repo-type dataset --include "data/*" --local-dir ./hn
echo "Done: $(find ./hn/data -name '*.parquet' | wc -l) monthly files in $(pwd)/hn"
