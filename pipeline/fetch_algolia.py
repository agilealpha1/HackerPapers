#!/usr/bin/env python3
"""Fetch every HN story/comment that links to a paper site, via the HN Algolia API.

Standard library only (runs on stock macOS python3). Resumable: re-run to continue.

  python3 pipeline/fetch_algolia.py                 # search phase (~150 requests, ~2 min)
  python3 pipeline/fetch_algolia.py --threads       # + full comment threads of paper stories
  python3 pipeline/fetch_algolia.py --query "hypernetwork" --query "weight space"
                                                    # query mode: search HN by text and append any new
                                                    # stories/comments that link a paper (then run extract_algolia.py)

Output: hn_algolia/stories.jsonl, comments.jsonl, threads.jsonl, fetch_log.json
How it works (Algolia returns at most 1,000 hits per query):
  for each (item type, field, domain term): count hits in a time window; if > 1,000,
  split the window in half and recurse; else fetch it in one page. Totals are checked
  against Algolia's nbHits so nothing is silently dropped.
"""
import argparse, json, os, ssl, subprocess, sys, time, urllib.parse, urllib.request

API = os.environ.get('HN_ALGOLIA_API', 'https://hn.algolia.com/api/v1')
UA = 'HackerPapers-pipeline/0.1'
HN_START = 1160000000          # 2006-10, before the first HN item
TERMS = ['arxiv', 'ar5iv', 'openreview', 'aclanthology', 'aclweb', 'nips', 'neurips', 'mlr']
QUERIES = [('story', 'url'), ('story', 'story_text'), ('comment', 'comment_text')]
FILES = {'story': 'stories.jsonl', 'comment': 'comments.jsonl'}
FIELDS = {'story': 'objectID,author,created_at_i,title,url,points,num_comments,story_text',
          'comment': 'objectID,author,created_at_i,comment_text,parent_id,story_id,story_title,story_url'}

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hn_algolia'))
ap.add_argument('--sleep', type=float, default=0.25)
ap.add_argument('--threads', action='store_true', help='also fetch full threads of paper stories')
ap.add_argument('--min-comments', type=int, default=1)
ap.add_argument('--since', type=int, default=HN_START, help='unix time; for incremental weekly runs')
ap.add_argument('--query', action='append', default=[], help='query mode: free-text HN search (repeatable)')
a = ap.parse_args()
OUT = os.path.abspath(a.out)
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, 'fetch_log.json')
log = json.load(open(LOG)) if os.path.exists(LOG) else {'done_windows': [], 'queries': {}, 'threads_done': []}
done = set(log['done_windows'])
use_curl = False
n_req = 0


def save_log():
    log['done_windows'] = sorted(done)
    tmp = LOG + '.tmp'
    json.dump(log, open(tmp, 'w'), indent=1)
    os.replace(tmp, LOG)


def get(path, params):
    global use_curl, n_req
    url = f'{API}/{path}?{urllib.parse.urlencode(params)}' if params else f'{API}/{path}'
    for attempt in range(8):
        n_req += 1
        time.sleep(a.sleep)
        try:
            if use_curl:
                body = subprocess.run(['curl', '-sS', '-f', '-m', '60', '-A', UA, url],
                                      check=True, capture_output=True).stdout
            else:
                req = urllib.request.Request(url, headers={'User-Agent': UA})
                body = urllib.request.urlopen(req, timeout=60).read()
            return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            wait = 60 if e.code == 429 else 2 ** attempt
            print(f'  HTTP {e.code}, retry in {wait}s', flush=True); time.sleep(wait)
        except (ssl.SSLError, urllib.error.URLError) as e:
            if isinstance(getattr(e, 'reason', e), ssl.SSLError) and not use_curl:
                print('  python SSL certs unavailable, switching to curl', flush=True)
                use_curl = True; continue
            print(f'  network error {e}, retry', flush=True); time.sleep(2 ** attempt)
        except subprocess.CalledProcessError as e:
            print(f'  curl error {e.returncode}, retry', flush=True); time.sleep(2 ** attempt)
    sys.exit(f'giving up on {url}')


def search(tag, attr, term, lo, hi, hits_per_page):
    p = {'query': term, 'tags': tag, 'restrictSearchableAttributes': attr, 'typoTolerance': 'false',
         'attributesToHighlight': '[]', 'hitsPerPage': hits_per_page,
         'numericFilters': f'created_at_i>={lo},created_at_i<{hi}'}
    if hits_per_page:
        p['attributesToRetrieve'] = FIELDS[tag]
    return get('search_by_date', p)


def crawl(tag, attr, term, lo, hi, sink):
    key = f'{tag}|{attr}|{term}|{lo}|{hi}'
    if key in done:
        return
    n = search(tag, attr, term, lo, hi, 0)['nbHits']
    if n > 1000 and hi - lo > 1:
        mid = (lo + hi) // 2
        crawl(tag, attr, term, lo, mid, sink); crawl(tag, attr, term, mid, hi, sink)
    elif n:
        log.setdefault('leaf_counts', {})[key] = n
        r = search(tag, attr, term, lo, hi, 1000)
        for h in r['hits']:
            h['_q'] = f'{attr}:{term}'
            sink.write(json.dumps(h, ensure_ascii=False) + '\n')
        sink.flush()
    done.add(key); save_log()


if a.query:
    # query mode: relevance search over all text, keep hits that link a paper and that we don't have yet
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from normalize import extract
    have = set()
    for tag in FILES:
        p = os.path.join(OUT, FILES[tag])
        if os.path.exists(p):
            for line in open(p, encoding='utf-8'):
                have.add(json.loads(line)['objectID'])
    total_new = 0
    for q in a.query:
        for tag in FILES:
            p = {'query': q, 'tags': tag, 'hitsPerPage': 1000, 'attributesToRetrieve': FIELDS[tag], 'attributesToHighlight': '[]'}
            if a.since > HN_START:
                p['numericFilters'] = f'created_at_i>={a.since}'
            r = get('search', p) or {'hits': [], 'nbHits': 0}
            linked = new = 0
            with open(os.path.join(OUT, FILES[tag]), 'a', encoding='utf-8') as sink:
                for h in r['hits']:
                    txt = ' '.join(h.get(k) or '' for k in ('url', 'story_text', 'comment_text'))
                    if not any(True for _ in extract(txt)):
                        continue
                    linked += 1
                    if h['objectID'] in have:
                        continue
                    have.add(h['objectID']); h['_q'] = 'query:' + q
                    sink.write(json.dumps(h, ensure_ascii=False) + '\n'); new += 1
            total_new += new
            print(f'query {tag:7s} {q!r}: {r.get("nbHits", 0)} hits, {linked} link a paper, {new} new', flush=True)
    log.setdefault('query_runs', []).append({'t': int(time.time()), 'queries': a.query, 'new': total_new}); save_log()
    print(f'NEW_HITS {total_new}', flush=True)
    sys.exit(0)


now = (int(time.time()) // 86400 + 1) * 86400   # end of today UTC: stable window keys within a day
t0 = time.time()
expected = {}
for tag, attr in QUERIES:
    with open(os.path.join(OUT, FILES[tag]), 'a', encoding='utf-8') as sink:
        for term in TERMS:
            expected[f'{attr}:{term}'] = search(tag, attr, term, a.since, now, 0)['nbHits']
            crawl(tag, attr, term, a.since, now, sink)
            print(f'{tag:7s} {attr:12s} {term:12s} algolia={expected[f"{attr}:{term}"]:6d}', flush=True)
# completeness check: Algolia's grand total is an estimate (exhaustiveNbHits=false) for big queries,
# so compare against the sum of the exact per-window counts instead
for q in expected:
    attr, term = q.split(':')
    expected[q] = sum(n for k, n in log.get('leaf_counts', {}).items()
                      if k.split('|')[1] == attr and k.split('|')[2] == term and int(k.split('|')[3]) >= a.since)
have = {}
for tag in FILES:
    p = os.path.join(OUT, FILES[tag])
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            h = json.loads(line)
            if h['created_at_i'] >= a.since:
                have.setdefault(h['_q'], set()).add(h['objectID'])
short = {q: n - len(have.get(q, ())) for q, n in expected.items() if len(have.get(q, ())) < n}
log['last_check'] = {'expected': expected, 'short': short}; save_log()
print('completeness: OK, every query fully fetched' if not short else f'completeness: SHORT {short} (re-run to retry)', flush=True)
print(f'search phase done: {n_req} requests, {time.time()-t0:.0f}s', flush=True)

if a.threads:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from normalize import extract
    stories = {}
    for line in open(os.path.join(OUT, 'stories.jsonl'), encoding='utf-8'):
        h = json.loads(line)
        if (h.get('num_comments') or 0) >= a.min_comments and any(True for _ in extract(h.get('url') or '')):
            stories[int(h['objectID'])] = h
    tdone = set(log['threads_done'])
    todo = [s for s in sorted(stories) if s not in tdone]
    print(f'threads: {len(stories)} paper stories with >= {a.min_comments} comments, {len(todo)} to fetch '
          f'(~{len(todo)*max(a.sleep,0.36)/60:.0f} min; Algolia allows ~10k requests/hour)', flush=True)

    def flat(node, story_id, depth, out):
        for c in node.get('children') or []:
            if c.get('type') == 'comment' and c.get('author'):
                out.append({'objectID': str(c['id']), 'author': c['author'], 'created_at_i': c.get('created_at_i'),
                            'comment_text': c.get('text'), 'parent_id': c.get('parent_id'),
                            'story_id': story_id, 'depth': depth})
            flat(c, story_id, depth + 1, out)

    with open(os.path.join(OUT, 'threads.jsonl'), 'a', encoding='utf-8') as sink:
        for i, sid in enumerate(todo):
            item = get(f'items/{sid}', None)
            rows = []
            if item:
                flat(item, sid, 1, rows)
            for r in rows:
                sink.write(json.dumps(r, ensure_ascii=False) + '\n')
            sink.flush()
            tdone.add(sid)
            if i % 50 == 0 or i == len(todo) - 1:
                log['threads_done'] = sorted(tdone); save_log()
                print(f'  {i+1}/{len(todo)} threads', flush=True)
        log['threads_done'] = sorted(tdone); save_log()
print(f'all done: {n_req} requests in {time.time()-t0:.0f}s -> {OUT}')
