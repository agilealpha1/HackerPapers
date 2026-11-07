#!/usr/bin/env python3
"""Step 4: bge-small-en-v1.5 embeddings for every arXiv paper with metadata (same model as HackerTube).

  python3 -m pip install --user fastembed numpy
  python3 pipeline/embed.py            # incremental: embeds only papers with no stored vector yet
  python3 pipeline/embed.py --all      # re-embed everything from scratch

Embeds "title. abstract" with no instruction prefix, L2-normalized, float16.
Writes data/meta/embeddings.npy and data/meta/embedding_ids.json (row i <-> id i).
build_site_data.py uses them for "related", "read first", For You and meaning-based search;
a paper with no vector gets a zero row in the site's emb.json and is invisible to those.
"""
import argparse, json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
EMB = os.path.join(ROOT, 'data', 'meta', 'embeddings.npy')
IDS = os.path.join(ROOT, 'data', 'meta', 'embedding_ids.json')
ap = argparse.ArgumentParser()
ap.add_argument('--all', action='store_true', help='re-embed every paper instead of only new ones')
ap.add_argument('--batch', type=int, default=64)
a = ap.parse_args()

meta = {}
for l in open(os.path.join(ROOT, 'data', 'meta', 'arxiv.jsonl'), encoding='utf-8'):
    m = json.loads(l); meta[m['paper_id']] = m          # last record wins if a paper was fetched twice
have_ids = [] if a.all or not os.path.exists(IDS) else json.load(open(IDS))
old = np.load(EMB).astype(np.float32) if have_ids else None
if old is not None and len(old) != len(have_ids):
    raise SystemExit(f'{EMB} has {len(old)} rows but {IDS} lists {len(have_ids)} ids: run with --all')
have = set(have_ids)
new = [m for pid, m in meta.items() if pid not in have]
print(f'{len(meta)} papers with metadata; {len(have)} already embedded; {len(new)} to embed', flush=True)
if not new:
    raise SystemExit(0)
from fastembed import TextEmbedding
model = TextEmbedding('BAAI/bge-small-en-v1.5')
E = np.array(list(model.embed([f"{m['title']}. {m['abstract']}" for m in new], batch_size=a.batch)), dtype=np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
if old is not None:
    E = np.vstack([old, E])
ids = have_ids + [m['paper_id'] for m in new]
tmp = EMB + '.tmp.npy'
np.save(tmp, E.astype(np.float16)); os.replace(tmp, EMB)
json.dump(ids, open(IDS + '.tmp', 'w')); os.replace(IDS + '.tmp', IDS)
print('done:', E.shape, flush=True)
