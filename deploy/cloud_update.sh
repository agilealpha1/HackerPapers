#!/usr/bin/env bash
# Stateless incremental update for GitHub Actions (or any machine without the full hn_algolia/ archive).
# Derives the 14-day sliding window from data/step1/mentions.jsonl, fetches recent HN stories/comments/threads,
# merges into data/step1/ via `extract_algolia.py --merge`, enriches new papers, and refreshes the weekly digest.
set -uo pipefail
cd "$(dirname "$0")/.."

INC_DIR="hn_algolia_inc"
trap 'rm -rf "$INC_DIR"' EXIT
rm -rf "$INC_DIR"

SINCE=$(python3 -c "import json, os; p='data/step1/mentions.jsonl'; ms=[json.loads(l)['time'] for l in open(p, encoding='utf-8') if l.strip()] if os.path.exists(p) else []; print(max(ms) - 14*86400 if ms else 1160000000)")
echo "== Incremental HN fetch since unix=$SINCE (14-day overlap)"

step() { echo "== $(date -u +%H:%M:%S) $*"; }
must() { step "$@"; "$@" || { echo "FAILED: $*"; exit 1; }; }
try()  { step "$@"; "$@" || echo "WARNING: $* failed; continuing with existing data"; }

must python3 -m unittest discover -s pipeline/tests
must python3 pipeline/fetch_algolia.py --out "$INC_DIR" --since "$SINCE"
try  python3 pipeline/fetch_algolia.py --out "$INC_DIR" --threads-only
must python3 pipeline/extract_algolia.py --in "$INC_DIR" --merge
must python3 pipeline/fetch_arxiv_meta.py
if [ "$((10#$(date -u +%d)))" -le 7 ]; then
  try python3 pipeline/fetch_s2.py --refresh --max-seconds 600
else
  try python3 pipeline/fetch_s2.py --max-seconds 600
fi
try  python3 pipeline/fetch_arxiv_meta.py
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  try python3 pipeline/label_llm.py --threads
else
  echo "== skipping label_llm.py: ANTHROPIC_API_KEY is not set (new comments use the v0 rules)"
fi
must python3 pipeline/build_site_data.py
try  python3 pipeline/fetch_hf.py --max-seconds 300
must python3 pipeline/build_site_data.py
try  python3 pipeline/weekly_digest.py
echo "== cloud_update done: $(date -u)"
