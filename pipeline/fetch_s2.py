#!/usr/bin/env python3
"""Step 3: Semantic Scholar citations, influential citations and venue for HN-recommended AI papers.

  python3 pipeline/fetch_s2.py                     # AI papers with >= 2 HN users (~4k), ~10 requests + title lookups
  S2_API_KEY=... python3 pipeline/fetch_s2.py      # optional key: higher rate limit
  python3 pipeline/fetch_s2.py --refresh           # also re-fetch citation counts for papers already done
                                                   # (batch endpoint only; the weekly job does this monthly)

Standard library only, resumable. Writes data/meta/s2.jsonl:
  {paper_id, s2_id, title, year, venue, citations, influential, arxiv_id, doi, url}
arXiv and ACL ids use the batch endpoint (500 per request); OpenReview/NeurIPS/PMLR papers are looked up by
title (one request each), which also gives their arXiv id when the paper has one.
"""
import csv, json, os, sys, time, urllib.parse, urllib.request, urllib.error

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
API = 'https://api.semanticscholar.org/graph/v1'
FIELDS = 'title,year,venue,citationCount,influentialCitationCount,externalIds,url'
KEY = os.environ.get('S2_API_KEY')
AI = {'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'cs.MA', 'stat.ML'}
OUT = os.path.join(ROOT, 'data', 'meta', 's2.jsonl')


def req(url, body=None):
    h = {'User-Agent': 'HackerPapers-pipeline/0.1', 'Content-Type': 'application/json'}
    if KEY:
        h['x-api-key'] = KEY
    for attempt in range(12):
        try:
            r = urllib.request.Request(url, data=json.dumps(body).encode() if body else None, headers=h)
            return json.loads(urllib.request.urlopen(r, timeout=90).read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            wait = min(30 * (attempt + 1), 180) if e.code in (429, 500, 502, 503, 504) else None
            if wait is None:
                print(f'  HTTP {e.code} for {url[:90]}', flush=True); return None
            print(f'  HTTP {e.code}, waiting {wait}s', flush=True); time.sleep(wait)
        except urllib.error.URLError as e:
            print(f'  network error {e}, retry', flush=True); time.sleep(5 * 2 ** attempt)
    print('  still rate-limited; stopping here. Re-run later (it resumes), or set S2_API_KEY.', flush=True)
    sys.exit(2)


def row(pid, p):
    ext = p.get('externalIds') or {}
    return dict(paper_id=pid, s2_id=p.get('paperId'), title=p.get('title'), year=p.get('year'), venue=p.get('venue') or '',
                citations=p.get('citationCount') or 0, influential=p.get('influentialCitationCount') or 0,
                arxiv_id=ext.get('ArXiv'), doi=ext.get('DOI'), url=p.get('url'))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='re-fetch counts for arXiv/ACL papers already in s2.jsonl')
    ap.add_argument('--max-seconds', type=int, default=0, help='stop cleanly after this many seconds (0 = no limit)')
    a = ap.parse_args()
    t0 = time.time()
    timed_out = lambda: bool(a.max_seconds and time.time() - t0 >= a.max_seconds)
    meta = {}
    for l in open(os.path.join(ROOT, 'data', 'meta', 'arxiv.jsonl'), encoding='utf-8'):
        m = json.loads(l); meta[m['paper_id']] = m
    papers = [p for p in csv.DictReader(open(os.path.join(ROOT, 'data', 'step1', 'papers_step1.csv'), encoding='utf-8'))
              if int(p['n_users']) >= 2 and (p['source'] != 'arxiv' or set(meta.get(p['paper_id'], {}).get('categories', [])) & AI)]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)['paper_id'] for l in open(OUT, encoding='utf-8')}
    todo = [p for p in papers if p['paper_id'] not in done]
    if a.refresh and os.path.exists(OUT):
        old = {}
        for l in open(OUT, encoding='utf-8'):
            r = json.loads(l); old[r['paper_id']] = r
        ids = [p['paper_id'] for p in papers if p['source'] in ('arxiv', 'acl') and old.get(p['paper_id'], {}).get('s2_id')]
        print(f'refresh: {len(ids)} papers already on Semantic Scholar', flush=True)
        try:
            for i in range(0, len(ids), 500 if KEY else 100):
                if timed_out():
                    print('  time budget reached during refresh; saving progress', flush=True); break
                chunk = ids[i:i + (500 if KEY else 100)]
                s2ids = [('ArXiv:' + x[6:]) if x.startswith('arxiv:') else ('ACL:' + x[4:]) for x in chunk]
                for pid, p in zip(chunk, req(f'{API}/paper/batch?fields={FIELDS}', {'ids': s2ids}) or []):
                    if p:
                        old[pid] = row(pid, p)
                time.sleep(1.1 if KEY else 6.0)
        finally:
            with open(OUT + '.tmp', 'w', encoding='utf-8') as f:
                for r in old.values():
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            os.replace(OUT + '.tmp', OUT)
        done = set(old)
    batch_ids = [p['paper_id'] for p in todo if p['source'] in ('arxiv', 'acl')]
    title_ids = [p for p in todo if p['source'] not in ('arxiv', 'acl')]
    print(f'{len(papers)} AI papers; {len(batch_ids)} by id (batch), {len(title_ids)} by title', flush=True)
    pause = 1.1 if KEY else 6.0
    with open(OUT, 'a', encoding='utf-8') as f:
        step = 500 if KEY else 100
        for i in range(0, len(batch_ids), step):
            if timed_out():
                print('  time budget reached; resuming remaining batch items next run', flush=True); break
            ids = batch_ids[i:i + step]
            s2ids = [('ArXiv:' + x[6:]) if x.startswith('arxiv:') else ('ACL:' + x[4:]) for x in ids]
            res = req(f'{API}/paper/batch?fields={FIELDS}', {'ids': s2ids}) or []
            for pid, p in zip(ids, res):
                f.write(json.dumps(row(pid, p) if p else dict(paper_id=pid, s2_id=None), ensure_ascii=False) + '\n')
            f.flush()
            print(f'  batch {i + len(ids)}/{len(batch_ids)}', flush=True)
            time.sleep(pause)
        for j, p in enumerate(title_ids):
            if timed_out():
                print('  time budget reached; resuming remaining title items next run', flush=True); break
            title = (p.get('top_story_title') or '').split(' [')[0].strip()
            r = req(f'{API}/paper/search/match?query={urllib.parse.quote(title)}&fields={FIELDS}') if len(title) > 15 else None
            hit = (r or {}).get('data', [None])[0] if r else None
            f.write(json.dumps(row(p['paper_id'], hit) if hit else dict(paper_id=p['paper_id'], s2_id=None), ensure_ascii=False) + '\n')
            f.flush()
            if j % 20 == 0:
                print(f'  title {j + 1}/{len(title_ids)}', flush=True)
            time.sleep(pause)
    rows = [json.loads(l) for l in open(OUT, encoding='utf-8')]
    found = [r for r in rows if r.get('s2_id')]
    print(f'done: {len(found)}/{len(rows)} found on Semantic Scholar; median citations '
          f'{sorted(r["citations"] for r in found)[len(found) // 2] if found else 0}')


if __name__ == '__main__':
    main()
