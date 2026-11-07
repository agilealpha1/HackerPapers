#!/usr/bin/env python3
"""Step 7: open-data export, HackerTube-style.

  python3 pipeline/export_open_data.py [--site-data site/data] [--out site/data/download]

Writes papers.jsonl, mentions.jsonl, taxonomy.json, related.jsonl, DATA.md and hackerpapers-data.zip.
Usernames appear only in mentions.jsonl (as on HN itself); related.jsonl carries counts only.
"""
import argparse, glob, json, os, zipfile
from datetime import datetime, timezone

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ap = argparse.ArgumentParser()
ap.add_argument('--site-data', default=os.path.join(ROOT, 'site', 'data'))
ap.add_argument('--out', default=None)
a = ap.parse_args()
OUT = a.out or os.path.join(a.site_data, 'download')
os.makedirs(OUT, exist_ok=True)

papers = {}
for f in glob.glob(os.path.join(a.site_data, 'p', '*.json')):
    papers.update(json.load(open(f, encoding='utf-8')))
index = json.load(open(os.path.join(a.site_data, 'index.json'), encoding='utf-8'))
order = [r['id'] for r in index]
stats = json.load(open(os.path.join(a.site_data, 'stats.json'), encoding='utf-8'))
taxo = json.load(open(os.path.join(a.site_data, 'taxonomy.json'), encoding='utf-8'))

KEEP = ['id', 'title', 'year', 'published', 'authors', 'n_authors', 'cats', 'abstract', 'tldr', 'topics', 'level', 'cites', 'infl', 'venue', 'hf',
        'users', 'pos', 'strong', 'critical', 'months', 'points', 'first', 'last', 'score', 'trend', 'classic', 'trending',
        'code', 'code_src', 'quote', 'stances', 'prereqs', 'stories', 'paths', 'xp', 'pick']
with open(os.path.join(OUT, 'papers.jsonl'), 'w', encoding='utf-8') as fp, \
     open(os.path.join(OUT, 'mentions.jsonl'), 'w', encoding='utf-8') as fm, \
     open(os.path.join(OUT, 'related.jsonl'), 'w', encoding='utf-8') as fr:
    for pid in order:
        p = papers[pid]
        fp.write(json.dumps({k: p.get(k) for k in KEEP}, ensure_ascii=False) + '\n')
        for m in p.get('mentions', []):
            fm.write(json.dumps(dict(hn_id=m['id'], hn_url=f"https://news.ycombinator.com/item?id={m['id']}", paper_id=pid,
                                     author=m['a'], month=m['mo'], stance=m['st'], story_title=m['stt'], text=m['t']),
                                ensure_ascii=False) + '\n')
        fr.write(json.dumps(dict(id=pid, related=[dict(id=r['id'], shared_hn_people=r['shared'], text_similarity=r['sim'])
                                                  for r in p.get('related', [])]), ensure_ascii=False) + '\n')
json.dump(taxo, open(os.path.join(OUT, 'taxonomy.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

n_m = sum(len(p.get('mentions', [])) for p in papers.values())
DATA_MD = f"""# HackerPapers data

The AI research papers that Hacker News keeps recommending, exported {datetime.now(timezone.utc):%Y-%m-%d}
(HN data through {stats.get('data_through')}).

- {sum(1 for x in index if not x.get('xp')):,} papers, each recommended by at least 2 different HN people,
  plus {sum(1 for x in index if x.get('xp')):,} kept only for curated reading paths (`xp: true`: below that bar;
  `pick: true`: an editor's pick that HN hasn't linked yet)
- {n_m:,} HN comments about them (up to 12 per paper, one per person, recommendations first)
- {sum(len(a['topics']) for a in taxo['areas'])} topics in {len(taxo['areas'])} areas

## For AI agents: start here

| file | one row is | key | joins to |
|---|---|---|---|
| `papers.jsonl` | a paper, best-scored first | `id` | the hub |
| `mentions.jsonl` | an HN comment about a paper | `hn_id` + `paper_id` | `paper_id` -> `papers.id` |
| `related.jsonl` | a paper's related papers | `id` | `related[].id` -> `papers.id` |
| `taxonomy.json` | the topic tree | topic `id` | `papers.topics[]` |

Things to know:

- `id` is a canonical paper ID: `arxiv:<id>` (version stripped; old-style IDs keep their archive,
  e.g. `arxiv:cs/0112017`), `openreview:<forum id>`, `acl:<anthology id>`, `neurips:<hash>` or `neurips:n<number>`,
  `pmlr:<volume>/<key>`. arXiv links: `https://arxiv.org/abs/<id>`, PDF `https://arxiv.org/pdf/<id>`.
- `score` is the ranking signal. `pos` = distinct HN people whose mention was a recommendation (a `rec` or `strong_rec`
  comment, or submitting the paper to a story that reached 10+ points). `users` = all distinct people.
- **Stances** (`strong_rec`, `rec`, `reference` = neutral citation, `critical`, `author` = the commenter wrote the
  paper; never counts as a recommendation): {stats.get('llm_labeled', 0):,} of {stats.get('comments_with_text', 0):,} HN comments were labeled by
  Claude (stance, a verbatim quote and a one-line reason). The rest, mostly the newest, fall back to v0 rules (phrase
  patterns + VADER sentiment), which are noisier. `tldr` is the first sentences of the arXiv abstract, not a summary.
- `mentions.text`: HTML stripped, links replaced by `[PAPER]` (this paper) and `[LINK]` (anything else), cut at 700 chars.
- `code` is a GitHub/GitLab URL list; `code_src` says where it came from: `paper` (the arXiv comment or abstract),
  `hf` (the paper's Hugging Face page), `author` (posted on HN by the paper's author) or `hn` (posted next to the
  paper on HN and its name matches the paper; the least reliable).
- `prereqs`: earlier, well-known papers on the same topic ("read first"). Heuristic.
- `paths`: the reading paths a paper is on (`{{id, name, step, of}}`). Curated paths come from `pipeline/make_path.py`.
- `cites` / `infl` / `venue`: Semantic Scholar citation counts and venue; `hf`: top Hugging Face models, datasets
  and Spaces that cite the paper.
- Not included yet: embeddings.

## How it was made

1. Every HN story and comment linking arxiv.org, openreview.net, aclanthology.org, papers.nips.cc,
   proceedings.neurips.cc or proceedings.mlr.press was fetched from the HN Algolia API (time windows split
   below Algolia's 1,000-hit cap; counts checked against Algolia's exact per-window totals).
2. Links were normalized to one ID per paper. arXiv metadata came from the arXiv API.
3. Papers were kept if any arXiv category is cs.AI, cs.LG, cs.CL, cs.CV, cs.NE, cs.MA or stat.ML (venue papers
   count as AI) and at least 2 different people recommended them.
4. A comment counts as real commentary with at least 100 characters of its own text (rule from HackerTube).

## License

ODC-By 1.0. Credit HackerPapers. HN comments remain the work of their authors; arXiv metadata is from arXiv.
Not affiliated with Hacker News, Y Combinator or arXiv.
"""
open(os.path.join(OUT, 'DATA.md'), 'w', encoding='utf-8').write(DATA_MD)
emb_p = os.path.join(ROOT, 'data', 'meta', 'embeddings.npy')
extra = []
if os.path.exists(emb_p):
    import numpy as np
    E = np.load(emb_p); eids = json.load(open(os.path.join(ROOT, 'data', 'meta', 'embedding_ids.json')))
    pos = {x: i for i, x in enumerate(eids)}
    keep = [x for x in order if x in pos]
    np.save(os.path.join(OUT, 'embeddings.npy'), E[[pos[x] for x in keep]].astype(np.float16))
    json.dump(keep, open(os.path.join(OUT, 'embedding_ids.json'), 'w'))
    extra = ['embeddings.npy', 'embedding_ids.json']
    DATA_MD = DATA_MD.replace('- Not included yet: embeddings.',
        '- `embeddings.npy` + `embedding_ids.json`: float16 [n, 384], L2-normalized, BAAI/bge-small-en-v1.5 (fastembed, no\n'
        '  instruction prefix) over "title. abstract". Row i is paper `embedding_ids[i]`.')
    open(os.path.join(OUT, 'DATA.md'), 'w', encoding='utf-8').write(DATA_MD)
zp = os.path.join(OUT, 'hackerpapers-data.zip')
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    for n in ['papers.jsonl', 'mentions.jsonl', 'related.jsonl', 'taxonomy.json', 'DATA.md'] + extra:
        z.write(os.path.join(OUT, n), n)
print('export ->', OUT, {n: os.path.getsize(os.path.join(OUT, n)) for n in os.listdir(OUT)})
