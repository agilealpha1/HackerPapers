#!/usr/bin/env python3
"""Validate and merge agent-written label batches into data/labels/llm_labels.jsonl.

  python3 pipeline/merge_labels.py --batches <dir with batch_NN.jsonl inputs> --labels <dir with outputs> --out llm_labels.jsonl

Checks: every input key labeled once, stance in the allowed set, quote is a verbatim substring of the
comment (else dropped). Prints per-batch gaps so missing batches can be re-run.
"""
import argparse, glob, json, os, re

STANCES = {'strong_rec', 'rec', 'reference', 'critical', 'joke', 'author', 'spam'}
ap = argparse.ArgumentParser()
ap.add_argument('--batches', required=True)
ap.add_argument('--labels', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()

norm = lambda s: re.sub(r'\s+', ' ', s).strip()
merged, gaps, bad_lines, dropped_quotes, total_in = [], {}, 0, 0, 0
for inp in sorted(glob.glob(os.path.join(a.batches, 'batch_*.jsonl'))):
    name = os.path.basename(inp)
    src = {}
    for l in open(inp, encoding='utf-8'):
        r = json.loads(l); src[r['k']] = r
    total_in += len(src)
    got = {}
    outp = os.path.join(a.labels, name)
    if os.path.exists(outp):
        for l in open(outp, encoding='utf-8'):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                bad_lines += 1; continue
            if r.get('k') in src and r.get('stance') in STANCES and r['k'] not in got:
                got[r['k']] = r
            else:
                bad_lines += 1
    missing = [k for k in src if k not in got]
    if missing:
        gaps[name] = len(missing)
    for k, r in got.items():
        q = (r.get('quote') or '').strip().strip('"“”')
        if q and (norm(q) not in norm(src[k]['text']) or '[PAPER]' in q or '[LINK]' in q or len(q) > 300):
            q = ''; dropped_quotes += 1
        hn_id, pid = k.split('|', 1)
        merged.append(dict(hn_id=int(hn_id), paper_id=pid, stance=r['stance'], spam=r['stance'] == 'spam', quote=q,
                           why=(r.get('why') or '')[:140] if r['stance'] in ('rec', 'strong_rec') else '', easy=bool(r.get('easy'))))
with open(a.out, 'w', encoding='utf-8') as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
from collections import Counter
print(json.dumps(dict(inputs=total_in, labeled=len(merged), coverage=round(len(merged) / max(total_in, 1), 4),
                      bad_lines=bad_lines, quotes_dropped_not_verbatim=dropped_quotes,
                      stances=Counter(r['stance'] for r in merged), batches_with_gaps=gaps), indent=1))
