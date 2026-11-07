# HackerPapers

The AI research papers Hacker News keeps recommending, with what HN actually said about them.
It's a static site: JSON data plus one HTML page, with no server and no accounts. Prototype v1.2.

## What the site has

- **Feeds:** Top papers, All-time classics, Trending this year, Has code, Beginner-friendly,
  Lasting favorites, From the authors, Most debated.
- **Paper page:**
  - TL;DR (the first lines of the abstract), why HN recommends it, and a pull quote.
  - **The author on HN**: the author's own comments, with a link to their HN profile.
  - **The debate**: recommendations next to pushback.
  - **HN over time**: a yearly chart, plus a quote from the first year and one from two or more years later.
  - What HN said, with each comment linking to its HN thread.
  - Code, Semantic Scholar citations, Hugging Face links, talks from HackerTube, reading paths,
    "read first" papers and related papers.
- **Search:** title, authors, TL;DR, quote and the "why" line; then partial word matches; then words in the full
  abstracts (`site/data/words.json`, loaded only when needed); each step adds papers close in meaning.
- **Explore:** topics (46 in 8 areas), reading paths (8 tracks, oldest first), Ask the papers, and search by
  keyword plus meaning (bge-small embeddings).
- **Yours:** For You (topics plus similarity to saved and read papers, each with a reason), a reading list and
  "mark as read". All of this lives only in the browser's localStorage.
- **Keyboard:** j/k move, Enter opens, o opens the PDF, s saves, r marks as read, / searches, g h/f/l/t/p/a jumps
  to a section, ? lists the shortcuts.
- **Atom feed** (`site_dist/feed.xml`) and a **weekly digest** (newsletter Markdown plus X posts).

## Pipeline

Run everything from this folder. Steps marked *(net)* need internet, so run them in your Mac Terminal.
Everything is resumable.

| step | command | output |
|---|---|---|
| 1 *(net)* | `python3 pipeline/fetch_algolia.py` | `hn_algolia/` raw HN hits (~150 requests) |
| 1 | `python3 pipeline/extract_algolia.py` | `data/step1/` mentions + per-paper counts |
| 3 *(net)* | `python3 pipeline/fetch_arxiv_meta.py` | `data/meta/arxiv.jsonl` |
| 3 *(net)* | `python3 pipeline/fetch_s2.py` (faster with `S2_API_KEY`) | `data/meta/s2.jsonl` citations, venue. Covers 560+ of the site's papers; backfilled 2026-11-04 through the batch endpoint with backoff |
| 3 *(net)* | `python3 pipeline/fetch_hf.py` | `data/meta/hf.jsonl`: HF paper pages, models, datasets, and each paper's **GitHub repo** (`github`, `github_stars`, `github_added_by`) |
| 3 | `data/meta/hf_code.json` | one-off backfill of HF GitHub repos for site papers that had no code link (fetched 2026-11-04) |
| 4 *(net, once)* | `python3 pipeline/embed.py` (needs `pip install fastembed numpy`) | `data/meta/embeddings.npy` bge-small |
| 2 | labels: `data/labels/llm_labels.jsonl` | Claude stance, quote, why and easy flag per comment |
| — *(net, once)* | HackerTube data → `data/hackertube/` | talks matched to papers |
| 2–6 | `python3 pipeline/build_site_data.py` | `site/data/` (index, 64 shards, embeddings, paths) |
| 7 | `python3 pipeline/export_open_data.py` | `site/data/download/` open data + DATA.md |
| 8 | `python3 pipeline/make_static.py` | `site_dist/`, the deployable site, plus feed.xml |
| 8 | `python3 pipeline/weekly_digest.py` | `data/digest/DATE.md` + `DATE.x.txt` |
| paths *(net)* | `python3 pipeline/make_path.py --topic "Weight-space learning: networks that read or write weights" --seeds 1609.09106,2402.13144 --keywords "hypernetwork;weight space"` (use your own topic, arXiv ids and 3+ character keywords) | a curated reading path in `data/curated_paths.json`, plus `data/paths/<id>/` (candidates, judgments, report.md) |

To view the site locally: `cd site_dist && python3 -m http.server 8000`, then open http://localhost:8000.

The weekly refresh is `deploy/weekly_update.sh`. It fetches new HN items, rebuilds the site and the export, and
writes the digest. Set `HP_BASE_URL` to the real domain for feed and digest links.

## How papers get in

- **Links:** stories and comments linking arxiv.org, openreview.net, aclanthology.org, papers.nips.cc,
  proceedings.neurips.cc or proceedings.mlr.press, normalized to one ID per paper. Venue copies are merged into
  the arXiv version by title.
- **AI only:** an arXiv category in cs.AI, cs.LG, cs.CL, cs.CV, cs.NE, cs.MA or stat.ML, and at least 2 ML terms
  in the title and abstract.
- **The bar:** at least 2 different HN people recommend the paper. A recommendation is a Claude-labeled rec or
  strong rec, or submitting it to a story that reached 10+ points. Papers whose author discussed them on HN need
  one recommendation from someone else.
- **Labels:** Claude labeled 9,028 of the 10,971 comments on these papers, with a stance, a verbatim quote and a
  one-line reason. About 1,950 recent comments still use the older rule-based labels.
- **Score:** distinct recommenders, distinct people, recommendation months, top post points, strong
  recommendations and citations, minus criticism, plus a recency boost, all on log scales.

## Curated reading paths

`pipeline/make_path.py` builds a reading path from a topic and a few seed papers. It finds candidates already in our
HN data by embedding similarity (top 50), keywords in titles, abstracts and HN comments, Semantic Scholar references and
citations of the seeds, and papers linked in the same HN threads as a seed. It also searches HN again
(`fetch_algolia.py --query`). An LLM then judges each candidate on-topic or not, with a one-line reason; without
`ANTHROPIC_API_KEY` it writes `judge_todo.jsonl` for Claude in Cowork to answer. Papers HN has discussed come first,
oldest to newest, then seeds HN never linked, flagged as Editor's pick. Path papers below the 2-recommender bar
stay out of the feeds, search and For You, but get their own pages. Optional `data/paths/<id>/stages.json` groups a
path into stages; the site shows the key papers (seeds, the most discussed, one per stage) with "Show all".
Arguments are checked before anything runs (arXiv ids, keywords of 3+ characters, no placeholder topic), and a search
with more than 1,500 candidates stops before writing anything.

## Setup

`python3 -m pip install -r requirements.txt`. The fetch scripts need only the standard library. `build_site_data.py`
uses scikit-learn (TF-IDF) and vaderSentiment when they are installed; without them `pipeline/fallbacks.py` stands
in (numpy TF-IDF, a small sentiment word list), so the build and the tests still run, with slightly different counts
(537 vs 564 papers at last check). `HP_FALLBACKS=1` forces the fallbacks.

## Weekly job, tests and versioning

`deploy/weekly_update.sh` runs, in order: the build tests; new HN items (`fetch_algolia.py --since`); comment
threads (`--threads-only`: new paper stories, and threads re-fetched while the story is under 14 days old, so later
comments are caught); arXiv metadata; Semantic Scholar for new papers (a full citation refresh in the first week of
each month); arXiv metadata again, for venue links S2 just matched to an arXiv id; embeddings for new papers only
(`embed.py`; `--all` re-embeds everything); Claude labels for new comments
when `ANTHROPIC_API_KEY` is set (`label_llm.py --threads`; without a key new comments use the v0 rules); the build;
Hugging Face links for new papers; a second build; the export, the static site and the digest. Enrichment steps may
fail without stopping the job.

Tests: `python3 -m unittest discover -s pipeline/tests -v` builds a small hand-made dataset twice (with and without
scikit-learn/vaderSentiment) and checks the recommendation bar, the AI filter, stance fallback rules, venue merging,
badges, thread comments, curated paths, label counts, the search word index and the output schema. `python3 pipeline/test_normalize.py` checks link normalization.

The folder is a git repository (`git log`). Raw and re-fetchable data (`hn_algolia/`, HackerTube, embeddings) and
build output (`site/data/`, `site_dist/`) are not tracked.

## Sharing

`make_static.py` writes `site_dist/p/<slug>/` and `site_dist/path/<id>/`: small pages with the paper's title and
OpenGraph/Twitter tags, so links shared on HN, X or Slack show a preview, and that forward to the app. On the deployed
site each paper page has a "Copy share link" button. Set `HP_BASE_URL` to the real domain before building.

## Badge and tab rules

- **Lasting:** either
  - HN keeps recommending it: 4+ recommendation months spread over 3+ years, or
  - it's moderately persistent on HN (3+ months over 2+ years) and well cited: 100+ Semantic Scholar
    citations per year since publication.
- **Spike:** 60%+ of all HN mentions came in one month within a single year, and the paper has fewer than
  50 citations a year. A burst on a heavily cited paper isn't called hype.
- **Author on HN:** Claude labeled the comment as the author's, and the commenter identifies themselves.
- **Debated:** at least 2 pushback comments, or 1 pushback on a paper with 3+ recommenders.
- **Beginner-friendly:** a survey or tutorial, or HN commenters call it accessible.
- **Code:** the first match wins, in this order:
  1. a repo linked from the paper itself (its arXiv comment or abstract);
  2. a repo the author posted on HN;
  3. the paper's GitHub repo on Hugging Face (the successor to Papers with Code). It counts if a person added it
     on HF, or if HF imported it automatically and the repo name matches a title word or the title's acronym;
     that check rejects mistakes like `karpathy/llama2.c` for Chinchilla;
  4. an HN-posted repo whose name matches the paper.

  The page labels each source: "Code", "posted by the author", "via Hugging Face" or "linked on HN".

## Data sources and licenses

HN data comes via the HN Algolia API. Paper metadata is from arXiv, citations from Semantic Scholar, and paper
pages and GitHub repos from Hugging Face. Semantic Scholar rate-limits anonymous use heavily. For a full
citation backfill, get a free key (https://www.semanticscholar.org/product/api#api-key-form) and run
`S2_API_KEY=... python3 pipeline/fetch_s2.py`. Talks are from HackerTube (ODC-By). The export is released under ODC-By 1.0. Not
affiliated with Hacker News, Y Combinator, arXiv or HackerTube.

## Folders

- `pipeline/`: all scripts.
- `data/`: inputs and intermediate files.
- `hn_algolia/`: raw HN hits.
- `site/`: the page and its built data.
- `site_dist/`: the deployable copy.
- `deploy/`: Cloudflare Pages headers and the weekly job.
- `_build/`: local staging for publishing (zips, temporary copies). It's git-ignored and never deployed, and
  safe to delete at any time.
