#!/usr/bin/env python3
"""Make a curated reading path for any topic from a few seed papers.

  python3 pipeline/make_path.py --topic "Weight-space learning: networks that read, write or generate weights" \
      --seeds 1609.09106,2402.13144 --keywords "hypernetwork;weight space;model zoo"

Arguments are checked first: seeds must be arXiv ids, keywords at least 3 characters, and the topic not a
placeholder. A search that finds more than --max-candidates (1,500) stops before writing anything.

Stages (default: all three in a row; each can be run alone with --stage):
  search  1. look the seeds up in our HN data (data/step1/mentions.jsonl and hn_algolia/)
          2. network (skip with --offline): arXiv metadata for the seeds, Semantic Scholar references and
             citations of each seed, and an HN Algolia search for the keywords, seed titles and seed ids
             (fetch_algolia.py --query); new hits go through extract_algolia.py
          3. candidates already in our HN data, from: embedding similarity to the seeds (bge-small, top --top),
             keyword hits in titles, abstracts and HN comments, S2 citation links, and papers linked in the
             same HN threads as a seed
  judge   4. on-topic yes/no with a one-line reason per candidate. With ANTHROPIC_API_KEY set this calls
             Claude (model: HP_JUDGE_MODEL, default claude-haiku-4-5). Without a key it writes
             judge_todo.jsonl and stops: put answers in judgments.jsonl ({"id","on_topic","reason"}),
             for example by asking Claude in Cowork, then re-run with --stage apply.
  apply   5. write the path into data/curated_paths.json: HN-mentioned papers first, oldest to newest,
             then seeds HN never mentioned, flagged as Editor's pick. Optional data/paths/<id>/stages.json
             ({"stages": [{"name", "blurb", "ids": [arXiv ids]}]}) groups the path into stages, in that order;
             within a stage the same rule applies. The --core most important steps (seeds first, then the
             most-discussed) are shown by default on the site, the rest behind "Show all".
          6. rebuild: build_site_data.py, export_open_data.py, make_static.py (skip with --no-rebuild)
          7. report: data/paths/<id>/report.md

Everything is cached in data/paths/<id>/, so re-runs only fetch and judge what is new.
Standard library + numpy. fastembed (pip install fastembed) embeds papers that have no stored
embedding yet; without it those papers are found by keywords, threads and citations only.
"""
import argparse, collections, json, os, re, subprocess, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser(description='Make a curated reading path for a topic from seed papers.')
ap.add_argument('--topic', required=True, help='"Name: one-line description" (the part before ":" names the path)')
ap.add_argument('--seeds', required=True, help='comma-separated arXiv ids')
ap.add_argument('--keywords', default='', help='semicolon-separated phrases for the keyword and HN searches')
ap.add_argument('--id', default=None, help='path id (default: slug of the name)')
ap.add_argument('--name', default=None)
ap.add_argument('--blurb', default=None)
ap.add_argument('--stage', default='all', choices=['all', 'search', 'judge', 'apply'])
ap.add_argument('--top', type=int, default=50, help='nearest papers by embedding')
ap.add_argument('--max-steps', type=int, default=40)
ap.add_argument('--core', type=int, default=12, help='steps shown before "Show all": the seeds, then the most-discussed on HN')
ap.add_argument('--offline', action='store_true', help='no network: use cached S2/Algolia results only')
ap.add_argument('--no-rebuild', action='store_true')
ap.add_argument('--s2-pages', type=int, default=3, help='pages of 1,000 citations to read per seed')
ap.add_argument('--max-candidates', type=int, default=1500, help='stop before judging if the search finds more (keywords too broad)')
a = ap.parse_args()

# ---- input checks: catch placeholders and typos before anything touches the network or the data
ARXIV_ID = re.compile(r'^(?:\d{4}\.\d{4,5}|[a-z][a-z.-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$')
_errs = []
_seeds = [s.strip().replace('arxiv:', '') for s in a.seeds.split(',') if s.strip()]
_bad = [s for s in _seeds if not ARXIV_ID.match(s)]
if not _seeds:
    _errs.append('--seeds is empty')
if _bad:
    _errs.append(f'--seeds: not arXiv ids: {", ".join(_bad)} (expected e.g. 1609.09106 or cs/0112017)')
_short = [k.strip() for k in a.keywords.split(';') if k.strip() and len(k.strip()) < 3]
if _short:
    _errs.append(f'--keywords: terms must be at least 3 characters: {", ".join(repr(k) for k in _short)}')
_name = (a.name or a.topic.split(':')[0]).strip()
if len(_name) < 3 or a.topic.strip().lower() in ('name: description', 'name', 'topic') or _name.lower() in ('name', 'topic', 'title'):
    _errs.append(f'--topic looks like a placeholder: {a.topic!r} (use "Path name: one-line description")')
if _errs:
    sys.exit('make_path.py: ' + '\n  '.join(['invalid arguments, nothing was run:'] + _errs))

name = a.name or a.topic.split(':')[0].strip()
blurb = a.blurb or (a.topic.split(':', 1)[1].strip() if ':' in a.topic else a.topic)
blurb = blurb[:1].upper() + blurb[1:]
PID = a.id or re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
SEEDS = list(dict.fromkeys('arxiv:' + s.strip().replace('arxiv:', '') for s in a.seeds.split(',') if s.strip()))
KW = [k.strip() for k in a.keywords.split(';') if k.strip()]
W = os.path.join(ROOT, 'data', 'paths', PID)
os.makedirs(W, exist_ok=True)
META_P = os.path.join(ROOT, 'data', 'meta', 'arxiv.jsonl')
MENT_P = os.path.join(ROOT, 'data', 'step1', 'mentions.jsonl')
EMB_P = os.path.join(ROOT, 'data', 'meta', 'embeddings.npy')
EMBI_P = os.path.join(ROOT, 'data', 'meta', 'embedding_ids.json')
CUR_P = os.path.join(ROOT, 'data', 'curated_paths.json')


def jl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()] if os.path.exists(p) else []


def wj(p, obj):
    tmp = p + '.tmp'
    json.dump(obj, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def say(*x):
    print(*x, flush=True)


def load_mentions():
    by = collections.defaultdict(list)
    for m in jl(MENT_P):
        by[m['paper_id']].append(m)
    return by


def load_meta():
    return {m['paper_id']: m for m in jl(META_P)}


# ---------------------------------------------------------------- network helpers
def http_json(url, headers=None, tries=8):
    h = {'User-Agent': 'HackerPapers-pipeline/0.1'}
    h.update(headers or {})
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=90).read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            wait = min(120, 5 * 2 ** i) if e.code == 429 else 2 ** i
            say(f'  HTTP {e.code}, retry in {wait}s'); time.sleep(wait)
        except Exception as e:  # URLError, SSL, timeouts
            if 'CERTIFICATE' in str(e).upper():
                try:   # stock macOS python often lacks certs: fall back to curl
                    return json.loads(subprocess.run(['curl', '-sS', '-f', '-m', '90', '-A', h['User-Agent']]
                                                     + [x for k, v in h.items() if k != 'User-Agent' for x in ('-H', f'{k}: {v}')]
                                                     + [url], check=True, capture_output=True).stdout)
                except Exception as e2:
                    e = e2
            say(f'  network error {e}, retry'); time.sleep(2 ** i)
    say(f'  giving up on {url}')
    return None


def fetch_arxiv(ids):
    """Append arXiv metadata for ids not in arxiv.jsonl yet (reuses fetch_arxiv_meta)."""
    meta = load_meta()
    todo = sorted({i for i in ids if i.startswith('arxiv:') and i not in meta})
    if not todo:
        return 0
    import fetch_arxiv_meta as fam
    got = []
    for k in range(0, len(todo), 100):
        got += fam.fetch_ids(todo[k:k + 100], 3.1); time.sleep(3.1)
    with open(META_P, 'a', encoding='utf-8') as f:
        for r in got:
            if r['paper_id'] not in meta:
                f.write(json.dumps(r, ensure_ascii=False) + '\n'); meta[r['paper_id']] = r
    say(f'  arXiv metadata: {len(got)} of {len(todo)} fetched')
    return len(got)


def embed_missing():
    """Embed papers in arxiv.jsonl that have no stored vector yet (needs fastembed)."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        say('  fastembed not installed: papers without a stored embedding are skipped by the similarity search')
        return 0
    meta = jl(META_P)
    ids = json.load(open(EMBI_P)) if os.path.exists(EMBI_P) else []
    have = set(ids)
    new = [m for m in meta if m['paper_id'] not in have]
    if not new:
        return 0
    model = TextEmbedding('BAAI/bge-small-en-v1.5')
    E = np.array(list(model.embed([f"{m['title']}. {m['abstract']}" for m in new], batch_size=64)), dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    old = np.load(EMB_P).astype(np.float32) if os.path.exists(EMB_P) else np.zeros((0, E.shape[1]), np.float32)
    np.save(EMB_P, np.vstack([old, E]).astype(np.float16))
    json.dump(ids + [m['paper_id'] for m in new], open(EMBI_P, 'w'))
    say(f'  embedded {len(new)} new papers')
    return len(new)


def s2_links():
    """References and citations of each seed (arXiv ids only), cached in s2_links.json."""
    P = os.path.join(W, 's2_links.json')
    cache = json.load(open(P)) if os.path.exists(P) else {}
    key = os.environ.get('S2_API_KEY')
    hdr = {'x-api-key': key} if key else {}
    base = 'https://api.semanticscholar.org/graph/v1/paper/arXiv:'
    for s in SEEDS:
        aid = s[6:]
        if aid in cache or a.offline:
            continue
        out = {'refs': [], 'cites': []}
        ok = True
        for kind, field in (('references', 'citedPaper'), ('citations', 'citingPaper')):
            for page in range(a.s2_pages if kind == 'citations' else 1):
                r = http_json(f'{base}{aid}/{kind}?fields=externalIds&limit=1000&offset={page * 1000}', hdr)
                time.sleep(1.1 if key else 3.0)
                if r is None and page == 0:
                    ok = False
                if not r or not r.get('data'):
                    break
                for row in r['data']:
                    x = (row.get(field) or {}).get('externalIds') or {}
                    if x.get('ArXiv'):
                        out['refs' if kind == 'references' else 'cites'].append(x['ArXiv'])
                if 'next' not in r:
                    break
        if ok:
            cache[aid] = out
            wj(P, cache)
        say(f'  S2 {aid}: {len(out["refs"])} arXiv references, {len(out["cites"])} arXiv citations' + ('' if ok else ' (not found or failed; will retry next run)'))
    return cache


def algolia_search(queries):
    """fetch_algolia.py --query for keywords, seed titles and ids; re-extract mentions if anything is new."""
    P = os.path.join(W, 'algolia_done.json')
    done = set(json.load(open(P))) if os.path.exists(P) else set()
    todo = [q for q in queries if q not in done]
    if not todo or a.offline:
        return 0
    cmd = [sys.executable, os.path.join(HERE, 'fetch_algolia.py')] + [x for q in todo for x in ('--query', q)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(''.join('  ' + l + '\n' for l in r.stdout.splitlines()))
    if r.returncode:
        sys.stderr.write(r.stderr[-2000:]); say('  Algolia search failed; continuing with the data we have'); return 0
    m = re.search(r'NEW_HITS (\d+)', r.stdout)
    new = int(m.group(1)) if m else 0
    wj(P, sorted(done | set(todo)))
    if new:
        say(f'  {new} new HN items link a paper: re-extracting mentions')
        subprocess.run([sys.executable, os.path.join(HERE, 'extract_algolia.py')], check=True, capture_output=True)
    return new


# ---------------------------------------------------------------- keyword matching
PREFIX = re.compile(r'^(hyper|meta|multi|pre|self|text|cross|sub)(?=[a-z]{3,})')


def kw_regex(k):
    parts = []
    for w in k.lower().split():
        w = PREFIX.sub(lambda m: m.group(1) + '[- ]?', re.escape(w))
        parts.append(w)
    return re.compile(r'\b' + r'[\s-]+'.join(parts) + r'(?:s|es)?\b', re.I)


# ---------------------------------------------------------------- stages 1-3: search
def search():
    rep = {'topic': a.topic, 'seeds': {}, 'when': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
    # 1. seeds in our HN data (raw hits: any line of hn_algolia/ that contains the id)
    raw = collections.Counter()
    for f in ('stories.jsonl', 'comments.jsonl', 'threads.jsonl'):
        p = os.path.join(ROOT, 'hn_algolia', f)
        if os.path.exists(p):
            for line in open(p, encoding='utf-8'):
                for s in SEEDS:
                    if s[6:] in line:
                        raw[s] += 1
    # 2. network
    if not a.offline:
        say('arXiv metadata for seeds…'); fetch_arxiv(SEEDS)
    meta = load_meta()
    if not a.offline:
        qs = KW + [meta[s]['title'] for s in SEEDS if s in meta] + [s[6:] for s in SEEDS]
        say(f'HN Algolia search: {len(qs)} queries…'); algolia_search(qs)
        say('Semantic Scholar references and citations…')
    links = s2_links()
    by = load_mentions()
    for s in SEEDS:
        ms = by.get(s, [])
        rep['seeds'][s] = dict(title=(meta.get(s) or {}).get('title'), hn_mentions=len(ms), hn_people=len({m['author'] for m in ms}),
                               raw_hits=raw[s], threads=sorted({m.get('story_title') or '' for m in ms})[:6])
    say('\n1. Seeds in our HN data')
    for s, r in rep['seeds'].items():
        say(f"  {s[6:]:12s} {('HN x%d' % r['hn_mentions']) if r['hn_mentions'] else 'not on HN':10s} {r['title'] or '(no metadata yet)'}")

    # 3. candidates (each must be linked on HN at least once)
    cand = collections.defaultdict(lambda: {'src': {}})
    for s, r in links.items():
        for kind, ids in (('ref', r['refs']), ('cites', r['cites'])):
            for x in ids:
                cid = 'arxiv:' + x
                if cid in by and cid not in SEEDS:
                    cand[cid]['src'].setdefault('s2', []).append(f'{"reference of" if kind == "ref" else "cites"} {s}')
    # co-mentioned: other papers linked in the same HN threads as a seed
    seed_threads = {m.get('story_id') for s in SEEDS for m in by.get(s, []) if m.get('story_id')}
    for cid, ms in by.items():
        if cid in SEEDS:
            continue
        th = {m.get('story_id') for m in ms} & seed_threads
        if th:
            cand[cid]['src']['thread'] = sorted({m.get('story_title') or '' for m in ms if m.get('story_id') in th})[:3]
    # keywords: titles and abstracts, and the HN comments and thread titles that link the paper
    rx = [(k, kw_regex(k)) for k in KW]
    for cid, ms in by.items():
        if cid in SEEDS:
            continue
        md = meta.get(cid) or {}
        paper_txt = (md.get('title') or '') + ' ' + (md.get('abstract') or '')
        hn_txt = ' '.join((m.get('text') or '') + ' ' + (m.get('story_title') or '') for m in ms)
        hit = [f'{k} (paper)' for k, r in rx if r.search(paper_txt)] + [f'{k} (HN)' for k, r in rx if not r.search(paper_txt) and r.search(hn_txt)]
        if hit:
            cand[cid]['src']['kw'] = hit
    # earlier yes-judgments stay candidates, so a re-run can't silently drop a paper from the path
    for r in jl(os.path.join(W, 'judgments.jsonl')):
        if r.get('on_topic') and r['id'] in by and r['id'] not in SEEDS:
            cand[r['id']]['src'].setdefault('kept', 'judged on-topic in an earlier run')
    # papers found via threads, S2 or HN text may lack metadata: fetch it, then embed anything new
    if not a.offline:
        need = [c for c in cand if c not in meta and c.startswith('arxiv:')]
        if need:
            say(f'arXiv metadata for {len(need)} candidates…'); fetch_arxiv(need)
        say('embeddings for new papers…'); embed_missing()
        meta = load_meta()
    # embedding similarity (max cosine to any seed), over papers linked on HN
    if os.path.exists(EMB_P):
        E = np.load(EMB_P).astype(np.float32)
        ids = json.load(open(EMBI_P))
        row = {p: i for i, p in enumerate(ids)}
        S = [row[s] for s in SEEDS if s in row]
        say(f'embedding similarity from {len(S)} of {len(SEEDS)} seeds' + ('' if len(S) == len(SEEDS) else ' (the rest have no stored embedding yet)'))
        if S:
            sim = (E @ E[S].T).max(axis=1)
            order = [i for i in np.argsort(-sim) if ids[i] in by and ids[i] not in SEEDS][:a.top]
            for i in order:
                cand[ids[i]]['src']['emb'] = round(float(sim[i]), 3)
            for cid in cand:
                if cid in row and 'emb' not in cand[cid]['src']:
                    cand[cid]['sim'] = round(float(sim[row[cid]]), 3)
    out = []
    for cid, c in cand.items():
        md, ms = meta.get(cid) or {}, by.get(cid, [])
        subs = [m for m in ms if m['kind'] == 'submission']
        title = md.get('title') or (max(subs, key=lambda m: m.get('points') or 0)['story_title'] if subs else None)
        if not title:
            continue
        out.append(dict(id=cid, title=title, published=(md.get('published') or '')[:10], year=int((md.get('published') or ms[0]['month'])[:4]),
                        hn_mentions=len(ms), hn_people=len({m['author'] for m in ms}), sources=c['src'],
                        sim=c['src'].get('emb', c.get('sim')), abstract=(md.get('abstract') or '')[:1400]))
    out.sort(key=lambda c: (-len(c['sources']), -(c['sim'] or 0)))
    if len(out) > a.max_candidates:
        sys.exit(f'{len(out)} candidates is more than --max-candidates {a.max_candidates}: the keywords are probably too broad. '
                 f'Nothing was written. Use more specific keywords, or raise --max-candidates.')
    with open(os.path.join(W, 'candidates.jsonl'), 'w', encoding='utf-8') as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    n_src = collections.Counter(k for c in out for k in c['sources'])
    rep['candidates'] = len(out); rep['by_source'] = dict(n_src)
    wj(os.path.join(W, 'search.json'), rep)
    say(f'\n2-3. {len(out)} candidates: ' + ', '.join(f'{k} {v}' for k, v in n_src.items()))
    return out


# ---------------------------------------------------------------- stage 4: judge
JUDGE_SYS = """You decide whether research papers belong on a reading path about one topic.
Topic: {topic}
on_topic is true only if the paper's main contribution is about this topic, not a passing use of a related technique.
Answer with only a JSON array, one object per paper, in order: [{{"i":0,"on_topic":true,"reason":"at most 18 words"}}]"""


def judge(cands):
    P = os.path.join(W, 'judgments.jsonl')
    got = {r['id']: r for r in jl(P)}
    todo = [c for c in cands if c['id'] not in got]
    if not todo:
        say(f'\n4. all {len(cands)} candidates judged'); return got
    key = os.environ.get('ANTHROPIC_API_KEY')
    if not key:
        with open(os.path.join(W, 'judge_todo.jsonl'), 'w', encoding='utf-8') as f:
            for c in todo:
                f.write(json.dumps(dict(id=c['id'], title=c['title'], year=c['year'], abstract=c['abstract'][:900]), ensure_ascii=False) + '\n')
        say(f'\n4. {len(todo)} candidates need a judgment and ANTHROPIC_API_KEY is not set.\n'
            f'   Wrote {os.path.relpath(os.path.join(W, "judge_todo.jsonl"), ROOT)}. Add one line per paper to\n'
            f'   {os.path.relpath(P, ROOT)} as {{"id": "...", "on_topic": true, "reason": "..."}} (Claude in Cowork can do this),\n'
            f'   then run the same command again with --stage apply.')
        return None
    model = os.environ.get('HP_JUDGE_MODEL', 'claude-haiku-4-5')
    for k in range(0, len(todo), 20):
        chunk = todo[k:k + 20]
        user = '\n\n'.join(f'[{i}] {c["title"]} ({c["year"]})\n{c["abstract"][:900]}' for i, c in enumerate(chunk))
        body = dict(model=model, max_tokens=3000, system=JUDGE_SYS.format(topic=a.topic), messages=[dict(role='user', content=user)])
        req = urllib.request.Request('https://api.anthropic.com/v1/messages', data=json.dumps(body).encode(),
                                     headers={'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'})
        txt = json.loads(urllib.request.urlopen(req, timeout=180).read())['content'][0]['text']
        arr = json.loads(txt[txt.index('['): txt.rindex(']') + 1])
        with open(P, 'a', encoding='utf-8') as f:
            for r in arr:
                c = chunk[int(r['i'])]
                row = dict(id=c['id'], on_topic=bool(r['on_topic']), reason=r.get('reason', ''), judge=model)
                got[c['id']] = row; f.write(json.dumps(row, ensure_ascii=False) + '\n')
        say(f'  judged {min(k + 20, len(todo))}/{len(todo)}')
    return got


# ---------------------------------------------------------------- stages 5-7: apply
def apply(cands, judged):
    meta, by = load_meta(), load_mentions()
    cmap = {c['id']: c for c in cands}
    keep = [c for c in cands if (judged.get(c['id']) or {}).get('on_topic')]
    hn_seeds = [s for s in SEEDS if by.get(s)]
    picks = [s for s in SEEDS if not by.get(s)]

    def pub(i):
        return (meta.get(i) or {}).get('published') or (cmap.get(i) or {}).get('published') or '9999'
    hn_steps = sorted(set([c['id'] for c in keep] + hn_seeds), key=pub)
    if len(hn_steps) > a.max_steps:
        rank = {c['id']: (c['hn_people'], c['sim'] or 0) for c in keep}
        top = set(hn_seeds) | set(sorted([c['id'] for c in keep], key=lambda i: rank[i], reverse=True)[:a.max_steps - len(hn_seeds)])
        hn_steps = [i for i in hn_steps if i in top]
    missing = [s for s in picks if s not in meta]
    if missing:
        say(f'  no arXiv metadata yet for {", ".join(x[6:] for x in missing)} (run without --offline): left out of the path for now')
    picks = sorted([s for s in picks if s in meta], key=pub)

    def step(i, pick):
        ms = by.get(i, [])
        r = dict(id=i, hn=len(ms), people=len({m['author'] for m in ms}))
        if pick:
            r['pick'] = True
        if i in SEEDS:
            r['seed'] = True
        why = (judged.get(i) or {}).get('reason')
        if why:
            r['why'] = why
        return r
    steps = [step(i, False) for i in hn_steps] + [step(i, True) for i in picks]
    # core: the seeds, then the most-discussed papers
    rank = sorted(steps, key=lambda s: (not s.get('seed'), -s['people'], -s['hn']))
    for s in rank[:a.core]:
        s['core'] = True
    path = dict(id=PID, name=name, blurb=blurb, topic=a.topic, seeds=[s[6:] for s in SEEDS], keywords=KW,
                updated=datetime.now(timezone.utc).strftime('%Y-%m-%d'), steps=steps)
    SP = os.path.join(W, 'stages.json')
    if os.path.exists(SP):
        st = json.load(open(SP, encoding='utf-8'))['stages']
        where = {('arxiv:' + i.replace('arxiv:', '')): k for k, g in enumerate(st) for i in g['ids']}
        extra = [s for s in steps if s['id'] not in where]
        if extra:
            st = st + [dict(name='More', blurb='', ids=[])]
            say(f'  {len(extra)} steps are in no stage of stages.json: put under "More"')
        order = {s['id']: n for n, s in enumerate(steps)}
        for s in steps:
            s['stage'] = where.get(s['id'], len(st) - 1)
        steps.sort(key=lambda s: (s['stage'], bool(s.get('pick')), order[s['id']]))
        for k in range(len(st)):     # every stage shows at least its most-discussed paper by default
            grp = [s for s in steps if s['stage'] == k]
            if grp and not any(s.get('core') for s in grp):
                max(grp, key=lambda s: (s['people'], s['hn']))['core'] = True
        path['stages'] = [dict(name=g['name'], blurb=g.get('blurb', '')) for g in st]
    cur = json.load(open(CUR_P)) if os.path.exists(CUR_P) else []
    cur = [p for p in cur if p['id'] != PID] + [path]
    wj(CUR_P, cur)
    say(f'\n5. {len(path["steps"])} steps written to data/curated_paths.json as "{PID}"')

    if not a.no_rebuild:
        for s in ('build_site_data.py', 'export_open_data.py', 'make_static.py'):
            say(f'6. {s}…')
            r = subprocess.run([sys.executable, os.path.join(HERE, s)], capture_output=True, text=True)
            if r.returncode:
                sys.stderr.write(r.stdout[-2000:] + r.stderr[-3000:]); sys.exit(f'{s} failed')
            lines = [l for l in r.stdout.splitlines() if l.strip()]
            for l in ([l for l in lines if 'path' in l] if s == 'build_site_data.py' else lines[-1:]):
                say('   ' + l)

    rep = json.load(open(os.path.join(W, 'search.json'))) if os.path.exists(os.path.join(W, 'search.json')) else {}
    L = [f'# Reading path: {name}', '', f'Topic: {a.topic}', '', f'Updated {path["updated"]}. Seeds: {", ".join(path["seeds"])}.', '',
         '## Seeds in our HN data', '']
    for s in SEEDS:
        ms = by.get(s, [])
        L.append(f'- {s[6:]} {(meta.get(s) or {}).get("title", "")}: ' + (f'{len(ms)} HN mentions by {len({m["author"] for m in ms})} people' if ms else 'not mentioned on HN'))
    L += ['', '## Search', '', f'{len(cands)} candidates ({", ".join(f"{k}: {v}" for k, v in rep.get("by_source", {}).items())}); '
          f'{len(keep)} judged on-topic, {len(cands) - len(keep)} rejected.', '',
          '## The path', '', '| # | paper | year | HN mentions | people | note |', '|---|---|---|---|---|---|']
    for n, s in enumerate(path['steps'], 1):
        md = meta.get(s['id']) or {}
        t = md.get('title') or (cmap.get(s['id']) or {}).get('title', s['id'])
        note = "Editor's pick" if s.get('pick') else ('seed' if s.get('seed') else s.get('why', ''))
        L.append(f'| {n} | {t} ({s["id"][6:]}) | {(md.get("published") or "")[:4]} | {s["hn"]} | {s["people"]} | {note} |')
    L += ['', '## Rejected', '']
    for c in cands:
        j = judged.get(c['id']) or {}
        if not j.get('on_topic'):
            L.append(f'- {c["title"]} ({c["id"][6:]}): {j.get("reason", "not judged")}')
    open(os.path.join(W, 'report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    say(f'7. report: {os.path.relpath(os.path.join(W, "report.md"), ROOT)}\n')
    for n, s in enumerate(path['steps'], 1):
        md = meta.get(s['id']) or {}
        say(f'  {n:2d}. {(md.get("published") or "")[:4]}  HN x{s["hn"]:<3d} {"[Editor’s pick] " if s.get("pick") else ""}{md.get("title", s["id"])}')


if __name__ == '__main__':
    cands = search() if a.stage in ('all', 'search') else jl(os.path.join(W, 'candidates.jsonl'))
    if a.stage == 'search':
        sys.exit(0)
    judged = judge(cands)
    if judged is None or a.stage == 'judge':
        sys.exit(0)
    apply(cands, judged)
