#!/usr/bin/env python3
"""Weekly digest: the AI papers HN recommended in the last 7 days, as Markdown (newsletter) and X-post text.

  python3 pipeline/weekly_digest.py [--days 7] [--top 8] [--base-url https://your.domain/]

Reads data/step1/mentions.jsonl (+ LLM labels) and site/data/index.json. Writes
data/digest/YYYY-MM-DD.md and data/digest/YYYY-MM-DD.x.txt (one post per line, each <= 280 chars).
Papers are ranked by distinct people recommending them this week; papers new to HN are marked "new".
"""
import argparse, collections, json, os, re
from datetime import datetime, timezone

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ap = argparse.ArgumentParser()
ap.add_argument('--days', type=int, default=7)
ap.add_argument('--top', type=int, default=8)
ap.add_argument('--base-url', default=os.environ.get('HP_BASE_URL', 'https://hackerpapers.example/'))
a = ap.parse_args()
base = a.base_url.rstrip('/') + '/'

idx = {r['id']: r for r in json.load(open(os.path.join(ROOT, 'site', 'data', 'index.json'), encoding='utf-8'))}
labels = {}
lp = os.path.join(ROOT, 'data', 'labels', 'llm_labels.jsonl')
if os.path.exists(lp):
    for l in open(lp, encoding='utf-8'):
        r = json.loads(l); labels[(r['hn_id'], r['paper_id'])] = r
ms = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'step1', 'mentions.jsonl'), encoding='utf-8')]
now = max(m['time'] for m in ms)
since = now - a.days * 86400
# fallback for comments the LLM pass has not labeled yet
REC = re.compile(r"\b(highly recommend\w*|must[- ]read|worth (a |the )?read\w*|recommend\w*|great paper|good paper|excellent paper|well worth|you should read|check out)\b", re.I)
first_seen = {}
for m in ms:
    first_seen[m['paper_id']] = min(first_seen.get(m['paper_id'], m['time']), m['time'])

week = collections.defaultdict(lambda: {'people': set(), 'quote': None})
for m in ms:
    if m['time'] < since or m['paper_id'] not in idx:
        continue
    lab = labels.get((m['hn_id'], m['paper_id']))
    rule_rec = (not lab and m['kind'] == 'comment' and m.get('substantive') and REC.search(m.get('text') or ''))
    positive = (lab and lab['stance'] in ('rec', 'strong_rec')) or rule_rec or (m['kind'] == 'submission' and (m.get('points') or 0) >= 10)
    if positive:
        w = week[m['paper_id']]
        w['people'].add(m['author'])
        if lab and lab.get('quote') and not w['quote']:
            w['quote'] = (lab['quote'], m['author'], m['hn_id'])
period = 'this week' if a.days == 7 else f'in the last {a.days} days'
picks = sorted(week.items(), key=lambda kv: (-len(kv[1]['people']), -idx[kv[0]]['score']))[:a.top]

day = datetime.fromtimestamp(now, timezone.utc).strftime('%Y-%m-%d')
out = os.path.join(ROOT, 'data', 'digest'); os.makedirs(out, exist_ok=True)
md = [f'# HackerPapers weekly: {day}', '', f'The AI papers Hacker News recommended in the last {a.days} days.', '']
posts = [f'{period.capitalize()} on HN: {len(picks)} AI papers people recommended 🧵 {base}#trending']
if not picks:
    md.append('_No new recommendations this week._')
for i, (pid, w) in enumerate(picks, 1):
    r = idx[pid]
    new = ' (new to HN)' if first_seen[pid] >= since else ''
    url = f"{base}#paper-{r['slug']}"
    md += [f"## {i}. [{r['title']}]({url}){new}", '',
           f"{len(w['people'])} HN {'person' if len(w['people']) == 1 else 'people'} recommended it {period}; {r['pos']} all-time."]
    if r.get('why'):
        md.append(f"Why: {r['why']}")
    if w['quote']:
        q, who, hid = w['quote']
        md.append(f'> "{q}" ({who}, https://news.ycombinator.com/item?id={hid})')
    md.append('')
    line = f"{i}. {r['title']}" + (f" — {r['why']}" if r.get('why') else '')
    line = (line[:280 - len(url) - 2].rstrip() + '…') if len(line) + len(url) + 1 > 280 else line
    posts.append(f'{line} {url}')
md += ['---', f'Browse everything: {base} · Data: HN via the Algolia API, arXiv. Not affiliated with Hacker News or Y Combinator.']
open(os.path.join(out, f'{day}.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
open(os.path.join(out, f'{day}.x.txt'), 'w', encoding='utf-8').write('\n'.join(posts) + '\n')
print(f'{len(picks)} papers -> data/digest/{day}.md and {day}.x.txt')
