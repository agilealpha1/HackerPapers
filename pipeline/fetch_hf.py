#!/usr/bin/env python3
"""Step 3: Hugging Face models, datasets and Spaces that cite each arXiv paper.

  python3 pipeline/fetch_hf.py [--top 2000]        # papers in the site build, best-scored first
  HF_TOKEN=... python3 pipeline/fetch_hf.py        # optional token: higher rate limit

Standard library only, resumable. Uses the Hub's arXiv filter (repos whose card cites arxiv:<id>), 3 requests
per paper, top 5 of each kind by downloads/likes. Writes data/meta/hf.jsonl:
  {paper_id, hf_paper: bool, models: [{id, downloads, likes}], datasets: [...], spaces: [...]}
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
API = 'https://huggingface.co/api'
OUT = os.path.join(ROOT, 'data', 'meta', 'hf.jsonl')
TOKEN = os.environ.get('HF_TOKEN')


def get(url):
    h = {'User-Agent': 'HackerPapers-pipeline/0.1'}
    if TOKEN:
        h['Authorization'] = 'Bearer ' + TOKEN
    for attempt in range(7):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 404):
                return None
            wait = 30 * (attempt + 1) if e.code == 429 else 3 * 2 ** attempt
            print(f'  HTTP {e.code}, waiting {wait}s', flush=True); time.sleep(wait)
        except urllib.error.URLError as e:
            print(f'  network error {e}, retry', flush=True); time.sleep(3 * 2 ** attempt)
    sys.exit('giving up on ' + url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=2000)
    ap.add_argument('--sleep', type=float, default=0.4)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(ROOT, 'site', 'data', 'index.json'), encoding='utf-8'))
    ids = [r['id'] for r in idx if r['id'].startswith('arxiv:') and r['year'] >= 2016][:a.top]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)['paper_id'] for l in open(OUT, encoding='utf-8')}
    todo = [i for i in ids if i not in done]
    print(f'{len(ids)} arXiv papers from 2016+; {len(todo)} to fetch (~{len(todo) * 4 * a.sleep / 60:.0f} min)', flush=True)
    with open(OUT, 'a', encoding='utf-8') as f:
        for n, pid in enumerate(todo):
            aid = pid[6:]
            pj = get(f'{API}/papers/{aid}')
            rec = dict(paper_id=pid, hf_paper=pj is not None, github=(pj or {}).get('githubRepo'),
                       github_stars=(pj or {}).get('githubStars'), github_added_by=(pj or {}).get('githubRepoAddedBy'))
            time.sleep(a.sleep)
            for kind, sort in (('models', 'downloads'), ('datasets', 'downloads'), ('spaces', 'likes')):
                r = get(f'{API}/{kind}?filter=arxiv:{aid}&sort={sort}&direction=-1&limit=5') or []
                rec[kind] = [dict(id=x.get('id'), downloads=x.get('downloads'), likes=x.get('likes')) for x in r if x.get('id')]
                time.sleep(a.sleep)
            f.write(json.dumps(rec, ensure_ascii=False) + '\n'); f.flush()
            if n % 50 == 0:
                print(f'  {n + 1}/{len(todo)}', flush=True)
    rows = [json.loads(l) for l in open(OUT, encoding='utf-8')]
    print(f"done: {sum(r['hf_paper'] for r in rows)} HF paper pages, {sum(bool(r['models']) for r in rows)} with models, "
          f"{sum(bool(r['datasets']) for r in rows)} with datasets, {sum(bool(r['spaces']) for r in rows)} with Spaces")


if __name__ == '__main__':
    main()
