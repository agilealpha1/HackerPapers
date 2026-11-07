#!/usr/bin/env python3
"""Fetch arXiv metadata for HN-mentioned papers and count the AI ones.

  python3 pipeline/fetch_arxiv_meta.py               # papers with >= 2 distinct HN users (~70 requests, ~4 min)
  python3 pipeline/fetch_arxiv_meta.py --min-users 1 # every arXiv paper (~290 requests, ~15 min)

Standard library only. Resumable. arXiv asks for <= 1 request every 3 seconds.
Writes data/meta/arxiv.jsonl (title, authors, abstract, dates, categories, ...) and
data/step1/papers_ai_step1.csv + data/step1/stats_ai.json.

AI = any category (primary or cross-list) in AI_CATS. Non-arXiv IDs (OpenReview, NeurIPS,
PMLR, ACL) are ML/NLP venues and count as AI.
"""
import argparse, csv, json, os, re, ssl, subprocess, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

AI_CATS = {'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'cs.MA', 'stat.ML'}
AI_ADJACENT = {'cs.RO', 'cs.IR', 'cs.SD', 'eess.AS', 'eess.IV', 'cs.HC'}
NS = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
API = os.environ.get('ARXIV_API', 'https://export.arxiv.org/api/query')
UA = 'HackerPapers-pipeline/0.1'
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def entry_id(url):
    m = re.search(r'arxiv\.org/abs/(.+?)(v\d+)?$', url)
    return m.group(1) if m else None


def parse_feed(xml_bytes):
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall('a:entry', NS):
        aid = entry_id(e.findtext('a:id', '', NS))
        if not aid:
            continue
        cats = [c.get('term') for c in e.findall('a:category', NS)]
        prim = e.find('arxiv:primary_category', NS)
        doi = e.findtext('arxiv:doi', None, NS)
        out.append({
            'paper_id': 'arxiv:' + aid,
            'title': ' '.join(e.findtext('a:title', '', NS).split()),
            'authors': [a.findtext('a:name', '', NS) for a in e.findall('a:author', NS)],
            'abstract': ' '.join(e.findtext('a:summary', '', NS).split()),
            'published': e.findtext('a:published', None, NS),
            'updated': e.findtext('a:updated', None, NS),
            'primary_category': prim.get('term') if prim is not None else (cats[0] if cats else None),
            'categories': cats,
            'comment': e.findtext('arxiv:comment', None, NS),
            'journal_ref': e.findtext('arxiv:journal_ref', None, NS),
            'doi': doi,
        })
    return out


class BadRequest(Exception):
    pass


def get(url, use_curl=[False]):
    for attempt in range(6):
        try:
            if use_curl[0]:
                return subprocess.run(['curl', '-sS', '-f', '-m', '90', '-A', UA, url], check=True, capture_output=True).stdout
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=90).read()
        except urllib.error.HTTPError as e:
            if e.code == 400:
                raise BadRequest(url)
            print(f'  HTTP {e.code}, retry', flush=True); time.sleep(5 * 2 ** attempt)
        except (ssl.SSLError, urllib.error.URLError) as e:
            if isinstance(getattr(e, 'reason', e), ssl.SSLError) and not use_curl[0]:
                use_curl[0] = True; continue
            print(f'  error {e}, retry', flush=True); time.sleep(5 * 2 ** attempt)
        except subprocess.CalledProcessError as e:
            if e.returncode == 22 and b'400' in (e.stderr or b''):
                raise BadRequest(url)
            print(f'  curl error {e.returncode}, retry', flush=True); time.sleep(5 * 2 ** attempt)
    sys.exit('giving up on ' + url)


INVALID = []


def fetch_ids(ids, pause):
    """Fetch a batch; arXiv answers 400 for the whole batch if one id is malformed, so bisect."""
    q = urllib.parse.urlencode({'id_list': ','.join(x[6:] for x in ids), 'max_results': len(ids)})
    try:
        return parse_feed(get(f'{API}?{q}'))
    except BadRequest:
        if len(ids) == 1:
            print(f'  arXiv rejects id {ids[0]} (logged, skipped)', flush=True)
            INVALID.append(ids[0]); return []
        mid = len(ids) // 2
        left = fetch_ids(ids[:mid], pause); time.sleep(pause)
        return left + fetch_ids(ids[mid:], pause)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--papers', default=os.path.join(ROOT, 'data', 'step1', 'papers_step1.csv'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'meta', 'arxiv.jsonl'))
    ap.add_argument('--min-users', type=int, default=2)
    ap.add_argument('--batch', type=int, default=100)
    ap.add_argument('--sleep', type=float, default=3.1)
    ap.add_argument('--offline', action='store_true', help='skip fetching; just recompute AI counts')
    a = ap.parse_args()
    a.out, a.papers = os.path.abspath(a.out), os.path.abspath(a.papers)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    papers = list(csv.DictReader(open(a.papers, encoding='utf-8')))
    want = [p['paper_id'] for p in papers if p['source'] == 'arxiv' and int(p['n_users']) >= a.min_users]
    # venue links (OpenReview, NeurIPS, ACL, PMLR) whose Semantic Scholar record names an arXiv version:
    # fetch that arXiv record too, so build_site_data.py can canonicalize the paper to it
    s2p = os.path.join(os.path.dirname(a.out), 's2.jsonl')
    if os.path.exists(s2p):
        venue = {p['paper_id'] for p in papers if p['source'] != 'arxiv' and int(p['n_users']) >= a.min_users}
        extra = []
        for line in open(s2p, encoding='utf-8'):
            r = json.loads(line)
            if r['paper_id'] in venue and r.get('arxiv_id'):
                extra.append('arxiv:' + r['arxiv_id'])
        want += [x for x in dict.fromkeys(extra) if x not in set(want)]
    meta = {}
    if os.path.exists(a.out):
        for line in open(a.out, encoding='utf-8'):
            m = json.loads(line); meta[m['paper_id']] = m
    tried_path = a.out + '.tried'
    tried = set(open(tried_path).read().split()) if os.path.exists(tried_path) else set()
    todo = [] if a.offline else [p for p in want if p not in meta and p not in tried]
    print(f'{len(want)} arXiv papers with >= {a.min_users} users; {len(todo)} to fetch '
          f'({(len(todo) + a.batch - 1) // a.batch} requests, ~{len(todo) / a.batch * a.sleep / 60:.0f} min)', flush=True)
    with open(a.out, 'a', encoding='utf-8') as f, open(tried_path, 'a') as ft:
        for i in range(0, len(todo), a.batch):
            ids = todo[i:i + a.batch]
            got = fetch_ids(ids, a.sleep)
            for m in got:
                if m['paper_id'] not in meta:
                    meta[m['paper_id']] = m; f.write(json.dumps(m, ensure_ascii=False) + '\n')
            f.flush(); ft.write('\n'.join(ids) + '\n'); ft.flush()
            print(f'  {min(i + a.batch, len(todo))}/{len(todo)}  (+{len(got)})', flush=True)
            time.sleep(a.sleep)

    # AI filter + counts
    rows, missing = [], 0
    for p in papers:
        if p['source'] == 'arxiv':
            m = meta.get(p['paper_id'])
            if not m:
                if int(p['n_users']) >= a.min_users:
                    missing += 1
                continue
            cats = set(m['categories'])
            p.update(title=m['title'], primary_category=m['primary_category'], categories=' '.join(m['categories']),
                     published=(m['published'] or '')[:10],
                     is_ai=bool(cats & AI_CATS), ai_primary=m['primary_category'] in AI_CATS,
                     ai_adjacent=bool(cats & AI_ADJACENT) and not (cats & AI_CATS))
        else:
            p.update(title=p.get('top_story_title'), primary_category=p['source'], categories='', published='',
                     is_ai=True, ai_primary=True, ai_adjacent=False)
        rows.append(p)
    ai = [r for r in rows if r['is_ai']]
    out_csv = os.path.join(os.path.dirname(a.papers), 'papers_ai_step1.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(sorted(ai, key=lambda r: (-int(r['n_users']), -int(r['max_points']))))
    ge = lambda rs, k, n=2: sum(1 for r in rs if int(r[k]) >= n)
    stats = {
        'arxiv_papers_checked': sum(1 for r in rows if r['source'] == 'arxiv'), 'arxiv_ids_not_found': missing,
        'ai_papers_ge2_users': ge(ai, 'n_users'),
        'ai_papers_ge2_substantive_users': ge(ai, 'n_substantive_users'),
        'ai_papers_hackertube_bar_3users_3months': sum(1 for r in ai if int(r['n_substantive_users']) >= 3 and int(r['n_substantive_months']) >= 3),
        'ai_primary_ge2_users': sum(1 for r in ai if r['ai_primary'] and int(r['n_users']) >= 2),
        'ai_adjacent_only_ge2_users': sum(1 for r in rows if r['ai_adjacent'] and int(r['n_users']) >= 2),
        'non_ai_ge2_users': sum(1 for r in rows if not r['is_ai'] and int(r['n_users']) >= 2),
        'invalid_ids_this_run': INVALID,
        'ai_cats': sorted(AI_CATS),
    }
    json.dump(stats, open(os.path.join(os.path.dirname(a.papers), 'stats_ai.json'), 'w'), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
