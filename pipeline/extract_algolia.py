#!/usr/bin/env python3
"""Step 1 (Algolia path): turn fetched HN hits into paper mentions + per-paper counts.

  python3 pipeline/extract_algolia.py [--in hn_algolia] [--out data/step1]

Rules borrowed from HackerTube's DATA.md:
  - a comment counts as real commentary only with >= 100 chars of its own text
    (HTML stripped, links removed); shorter ones are kept but flagged
  - links in the stored text become [PAPER] (this paper) / [LINK] (anything else), text cut at 1,500 chars
  - recommenders are counted by distinct people, and by distinct months
Outputs: mentions.jsonl, thread_comments.jsonl (if threads were fetched), papers_step1.csv, stats.json, report.md
"""
import argparse, csv, html, json, os, re, sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import extract, normalize, clean_url, URL_RE, BARE_RE, ARXIV_TAG_RE

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ap = argparse.ArgumentParser()
ap.add_argument('--in', dest='inp', default=os.path.join(ROOT, 'hn_algolia'))
ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'step1'))
ap.add_argument('--min-own-chars', type=int, default=100)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

A_TAG = re.compile(r'<a\s[^>]*href="([^"]*)"[^>]*>.*?</a>', re.I | re.S)
TAG = re.compile(r'<[^>]+>')


def plain(htm):
    """HN comment HTML -> plain text with bare URLs (hrefs kept, anchor text dropped)."""
    if not htm:
        return ''
    t = A_TAG.sub(lambda m: ' ' + html.unescape(m.group(1)) + ' ', htm)
    t = re.sub(r'<p>', '\n\n', t, flags=re.I)
    t = html.unescape(TAG.sub('', t))
    return re.sub(r'[ \t]+', ' ', t).strip()


def mask(text, paper_id):
    """Replace links with [PAPER]/[LINK]; return (masked_text, own_char_count)."""
    def sub(m):
        u = clean_url(m.group(0))
        r = normalize(u)
        return ' [PAPER] ' if r and r[0] == paper_id else ' [LINK] '
    t = URL_RE.sub(sub, text)
    t = BARE_RE.sub(sub, t)
    t = ARXIV_TAG_RE.sub(lambda m: ' [PAPER] ' if 'arxiv:' + m.group(1) == paper_id else ' [LINK] ', t)
    t = re.sub(r'[ \t]+', ' ', t).strip()
    own = len(re.sub(r'\[(PAPER|LINK)\]|\s+', '', t))
    return t, own


def month(ts):
    return datetime.fromtimestamp(int(ts), timezone.utc).strftime('%Y-%m')


def read(name):
    p = os.path.join(a.inp, name)
    if not os.path.exists(p):
        return []
    seen, out = set(), []
    for line in open(p, encoding='utf-8'):
        h = json.loads(line)
        if h['objectID'] not in seen:
            seen.add(h['objectID']); out.append(h)
    return out


stories, comments, threads = read('stories.jsonl'), read('comments.jsonl'), read('threads.jsonl')
print(f'hits: {len(stories)} stories, {len(comments)} comments, {len(threads)} thread comments')

mentions = []
story_paper = {}                         # story id -> paper id (story URL is a paper)
for s in stories:
    sid = int(s['objectID'])
    if not s.get('author'):
        continue
    for pid, src, raw in extract(s.get('url') or ''):
        story_paper[sid] = pid
        mentions.append(dict(hn_id=sid, kind='submission', paper_id=pid, source=src, author=s['author'],
                             time=s['created_at_i'], month=month(s['created_at_i']), story_id=sid,
                             story_title=s.get('title'), points=s.get('points'), num_comments=s.get('num_comments'),
                             own_chars=None, substantive=False, raw_url=raw, text=None))
        break
    if s.get('story_text'):
        txt = plain(s['story_text'])
        for pid, src, raw in extract(txt):
            mt, own = mask(txt, pid)
            mentions.append(dict(hn_id=sid, kind='story_text', paper_id=pid, source=src, author=s['author'],
                                 time=s['created_at_i'], month=month(s['created_at_i']), story_id=sid,
                                 story_title=s.get('title'), points=s.get('points'), num_comments=s.get('num_comments'),
                                 own_chars=own, substantive=own >= a.min_own_chars, raw_url=raw, text=mt[:1500]))

for c in comments:
    if not c.get('author') or not c.get('comment_text'):
        continue
    txt = plain(c['comment_text'])
    for pid, src, raw in extract(txt):
        mt, own = mask(txt, pid)
        mentions.append(dict(hn_id=int(c['objectID']), kind='comment', paper_id=pid, source=src, author=c['author'],
                             time=c['created_at_i'], month=month(c['created_at_i']), story_id=c.get('story_id'),
                             story_title=c.get('story_title'), points=None, num_comments=None,
                             own_chars=own, substantive=own >= a.min_own_chars, raw_url=raw, text=mt[:1500]))

with open(os.path.join(a.out, 'mentions.jsonl'), 'w', encoding='utf-8') as f:
    for m in mentions:
        f.write(json.dumps(m, ensure_ascii=False) + '\n')

# discussion under paper submissions (no link needed) -> candidates for stance labeling in step 2
thread_rows = []
for t in threads:
    pid = story_paper.get(int(t['story_id']))
    if pid and t.get('author') and t.get('comment_text'):
        txt, own = mask(plain(t['comment_text']), pid)
        thread_rows.append(dict(hn_id=int(t['objectID']), paper_id=pid, story_id=int(t['story_id']), depth=t.get('depth'),
                                author=t['author'], time=t['created_at_i'], month=month(t['created_at_i']),
                                own_chars=own, substantive=own >= a.min_own_chars, text=txt[:1500]))
if thread_rows:
    with open(os.path.join(a.out, 'thread_comments.jsonl'), 'w', encoding='utf-8') as f:
        for r in thread_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# per-paper aggregation
P = defaultdict(lambda: dict(users=set(), sub_users=set(), months=set(), sub_months=set(), submitters=set(),
                             comment_linkers=set(), n=0, max_points=0, sum_points=0, title=None, title_pts=-1,
                             first=None, last=None, source=None, thread_users=set(), thread_sub_users=set()))
for m in mentions:
    p = P[m['paper_id']]
    p['source'] = m['source']; p['n'] += 1
    p['users'].add(m['author']); p['months'].add(m['month'])
    if m['kind'] == 'submission':
        p['submitters'].add(m['author'])
        pts = m['points'] or 0
        p['max_points'] = max(p['max_points'], pts); p['sum_points'] += pts
        if pts > p['title_pts']:
            p['title'], p['title_pts'] = m['story_title'], pts
    else:
        if m['kind'] == 'comment':
            p['comment_linkers'].add(m['author'])
        if m['substantive']:
            p['sub_users'].add(m['author']); p['sub_months'].add(m['month'])
    p['first'] = min(p['first'] or m['time'], m['time']); p['last'] = max(p['last'] or 0, m['time'])
for r in thread_rows:
    p = P[r['paper_id']]
    p['thread_users'].add(r['author'])
    if r['substantive']:
        p['thread_sub_users'].add(r['author'])

rows = []
for pid, p in P.items():
    rows.append(dict(paper_id=pid, source=p['source'], n_mentions=p['n'], n_users=len(p['users']),
                     n_submitters=len(p['submitters']), n_comment_linkers=len(p['comment_linkers']),
                     n_substantive_users=len(p['sub_users']), n_months=len(p['months']),
                     n_substantive_months=len(p['sub_months']), max_points=p['max_points'], sum_points=p['sum_points'],
                     n_thread_users=len(p['thread_users']), n_users_incl_threads=len(p['users'] | p['thread_users']),
                     n_substantive_incl_threads=len(p['sub_users'] | p['thread_sub_users']),
                     first_seen=month(p['first']), last_seen=month(p['last']), top_story_title=p['title']))
rows.sort(key=lambda r: (-r['n_users'], -r['max_points']))
with open(os.path.join(a.out, 'papers_step1.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['paper_id'])
    w.writeheader(); w.writerows(rows)


def cnt(pred):
    return sum(1 for r in rows if pred(r))


def by_src(pred):
    d = defaultdict(int)
    for r in rows:
        if pred(r):
            d[r['source']] += 1
    return dict(sorted(d.items(), key=lambda x: -x[1]))


ge2 = lambda r: r['n_users'] >= 2
ge2s = lambda r: r['n_substantive_users'] >= 2
stats = dict(
    hits_stories=len(stories), hits_comments=len(comments), hits_thread_comments=len(threads),
    mentions=len(mentions), mentions_by_kind={k: sum(1 for m in mentions if m['kind'] == k) for k in ('submission', 'story_text', 'comment')},
    substantive_comment_mentions=sum(1 for m in mentions if m['substantive']),
    unique_papers=len(rows), unique_papers_by_source=by_src(lambda r: True),
    papers_ge2_users=cnt(ge2), papers_ge2_users_by_source=by_src(ge2),
    papers_ge2_substantive_users=cnt(ge2s), papers_ge2_substantive_users_by_source=by_src(ge2s),
    papers_hackertube_bar_3users_3months=cnt(lambda r: r['n_substantive_users'] >= 3 and r['n_substantive_months'] >= 3),
)
if thread_rows:
    stats['papers_ge2_users_incl_threads'] = cnt(lambda r: r['n_users_incl_threads'] >= 2)
    stats['papers_ge2_substantive_incl_threads'] = cnt(lambda r: r['n_substantive_incl_threads'] >= 2)
json.dump(stats, open(os.path.join(a.out, 'stats.json'), 'w'), indent=2)
with open(os.path.join(a.out, 'report.md'), 'w', encoding='utf-8') as f:
    f.write('# Step 1 report (Algolia)\n\n```json\n' + json.dumps(stats, indent=2) + '\n```\n\n## Top 30 by distinct users\n\n'
            '| paper | users | substantive users | months | max points | title |\n|---|---|---|---|---|---|\n')
    for r in rows[:30]:
        f.write(f"| {r['paper_id']} | {r['n_users']} | {r['n_substantive_users']} | {r['n_months']} | {r['max_points']} | "
                f"{(r['top_story_title'] or '').replace('|', '/')} |\n")
print(json.dumps(stats, indent=2))
