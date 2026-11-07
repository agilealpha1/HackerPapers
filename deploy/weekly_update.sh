#!/usr/bin/env bash
# Weekly job: pull new HN mentions, enrich new papers, rebuild the site and the open-data export.
# Run from the HackerPapers folder (cron example: 17 3 * * 1  cd /path/HackerPapers && deploy/weekly_update.sh >> deploy/update.log 2>&1)
#
# Optional environment:
#   ANTHROPIC_API_KEY  label new HN comments with Claude (label_llm.py); without it new comments use the v0 rules
#   S2_API_KEY         faster Semantic Scholar calls;   HF_TOKEN  faster Hugging Face calls
#   HP_BASE_URL        the site's public URL (feed, digest and share pages)
# Steps that only enrich (labels, embeddings, citations, HF) are allowed to fail: the build still runs with what we have.
set -uo pipefail
cd "$(dirname "$0")/.."
LAST_FILE=hn_algolia/.last_run
SINCE=$(cat "$LAST_FILE" 2>/dev/null || echo 1160000000)
# re-read the last 14 days too: comments get edited and stories keep collecting points
SINCE=$((SINCE - 14*86400))
NOW=$(date +%s)
step() { echo "== $(date -u +%H:%M:%S) $*"; }
must() { step "$@"; "$@" || { echo "FAILED: $*"; exit 1; }; }
try()  { step "$@"; "$@" || echo "WARNING: $* failed; continuing with existing data"; }

python3 -c "import sklearn, vaderSentiment" 2>/dev/null || echo "WARNING: scikit-learn / vaderSentiment missing (pip install -r requirements.txt); the build uses pipeline/fallbacks.py and counts will differ slightly"
must python3 -m unittest discover -s pipeline/tests   # stop if the build rules broke
must python3 pipeline/fetch_algolia.py --since "$SINCE"
try  python3 pipeline/fetch_algolia.py --threads-only   # threads of new paper stories, re-fetched while < 14 days old (first run: long)
must python3 pipeline/extract_algolia.py
must python3 pipeline/fetch_arxiv_meta.py            # arXiv metadata: S2 needs the categories to pick AI papers
if [ "$(date +%d)" -le 7 ]; then
  try python3 pipeline/fetch_s2.py --refresh        # first run of the month: refresh citation counts too
else
  try python3 pipeline/fetch_s2.py                  # new papers only
fi
try  python3 pipeline/fetch_arxiv_meta.py            # again: arXiv versions of venue links S2 just matched
try  python3 pipeline/embed.py                      # incremental: only papers with no vector yet
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  try python3 pipeline/label_llm.py --threads       # only comments not labeled yet, thread comments too (Message Batches, resumable)
else
  echo "== skipping label_llm.py: ANTHROPIC_API_KEY is not set (new comments use the v0 rules)"
fi
must python3 pipeline/build_site_data.py            # pass 1: the index fetch_hf.py reads
try  python3 pipeline/fetch_hf.py                   # new papers only
must python3 pipeline/build_site_data.py            # pass 2: with HF links and code
must python3 pipeline/export_open_data.py
must python3 pipeline/make_static.py
try  python3 pipeline/weekly_digest.py
echo "$NOW" > "$LAST_FILE"
# Cloudflare Pages (needs `npm i -g wrangler` and CLOUDFLARE_API_TOKEN):
#   wrangler pages deploy site_dist --project-name hackerpapers
echo "updated: $(date -u)"
