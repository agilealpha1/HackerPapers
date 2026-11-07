#!/usr/bin/env python3
"""Steps 2-5 (prototype v0): label mentions, enrich papers, rank them, write the site's JSON.

  python3 pipeline/build_site_data.py [--root .] [--out site/data]

Inputs (from steps 1 and 3):
  data/step1/mentions.jsonl   one row per HN item linking a paper
  data/meta/arxiv.jsonl       arXiv metadata (title, authors, abstract, categories, ...)
  hn_algolia/comments.jsonl, stories.jsonl  raw hits (optional; used for GitHub links)

v0 is rule-based, so it runs with no API key:
  - stance: phrase rules + VADER sentiment -> strong_rec / rec / reference / critical / author (self-promo)
  - quote: best verbatim sentence from a positive comment
  - topics: keyword taxonomy (pipeline/taxonomy.py), 1-3 per paper
  - related: same HN people + TF-IDF similarity; "read first" = similar, earlier, well-known papers
  - difficulty, code links (arXiv comments/abstract, then GitHub links posted next to the paper on HN)
The LLM pass (step 2/4 proper) can later overwrite stance/quote/why/tldr per mention or paper.
"""
import argparse, collections, hashlib, html, json, math, os, re, sys
from datetime import datetime, timezone

import numpy as np
# scikit-learn and vaderSentiment are optional: pipeline/fallbacks.py stands in (HP_FALLBACKS=1 forces it)
_FORCE = os.environ.get('HP_FALLBACKS') == '1'
try:
    if _FORCE: raise ImportError
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None
try:
    if _FORCE: raise ImportError
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import TOPICS, AREAS
import fallbacks
if TfidfVectorizer is None:
    TfidfVectorizer = fallbacks.SimpleTfidf
if SentimentIntensityAnalyzer is None:
    SentimentIntensityAnalyzer = fallbacks.SimpleSentiment
print(f'TF-IDF: {TfidfVectorizer.__module__.split(".")[0]}; sentiment: {SentimentIntensityAnalyzer.__module__.split(".")[0]}', flush=True)

AI_CATS = {'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'cs.MA', 'stat.ML'}
N_SHARDS = 64

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ap.add_argument('--out', default=None)
ap.add_argument('--min-positive', type=int, default=2)
ap.add_argument('--topic-margin', type=float, default=0.04, help='embedding check: how much closer another topic must be')
a = ap.parse_args()
ROOT = os.path.abspath(a.root)
OUT = os.path.abspath(a.out or os.path.join(ROOT, 'site', 'data'))
os.makedirs(os.path.join(OUT, 'p'), exist_ok=True)


def jl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


mentions = jl(os.path.join(ROOT, 'data', 'step1', 'mentions.jsonl'))
meta = {m['paper_id']: m for m in jl(os.path.join(ROOT, 'data', 'meta', 'arxiv.jsonl'))}
print(f'{len(mentions)} mentions, {len(meta)} arXiv records', flush=True)
hf = {r['paper_id']: r for r in jl(os.path.join(ROOT, 'data', 'meta', 'hf.jsonl'))}
s2 = {r['paper_id']: r for r in jl(os.path.join(ROOT, 'data', 'meta', 's2.jsonl')) if r.get('s2_id')}
_hfc = json.load(open(os.path.join(ROOT, 'data', 'meta', 'hf_code.json'))) if os.path.exists(os.path.join(ROOT, 'data', 'meta', 'hf_code.json')) else {'human': [], 'repos': {}}
HF_CODE, HF_HUMAN = _hfc['repos'], set(_hfc['human'])
for r in hf.values():   # repos saved by newer fetch_hf.py runs
    if r.get('github') and r['paper_id'].startswith('arxiv:'):
        HF_CODE.setdefault(r['paper_id'][6:], [r['github'], r.get('github_stars') or 0])
        if r.get('github_added_by') not in (None, 'auto'):
            HF_HUMAN.add(r['paper_id'][6:])


def repo_matches(title, url):
    # repo owner/name shares a distinctive title word, or the title's acronym (e.g. VAR, LIMO, BLT)
    name = re.sub(r'[^a-z0-9]+', ' ', url.lower().split('github.com/', 1)[-1])
    toks = set(name.split()) | {name.replace(' ', '')}
    words = re.findall(r'[A-Za-z0-9][A-Za-z0-9-]*', title.split(':')[0])
    stop = {'with', 'from', 'that', 'this', 'learning', 'neural', 'model', 'models', 'language', 'large', 'networks',
            'deep', 'towards', 'using', 'training', 'what', 'your', 'need', 'scaling', 'data', 'paper', 'llms', 'llm'}
    tw = {w.lower().replace('-', '') for w in words if len(w) >= 4 and w.lower() not in stop}
    acro = ''.join(w[0] for w in words if w[0].isalpha()).lower()
    return bool(tw & {t.replace('-', '') for t in toks}) or (len(acro) >= 3 and any(acro == t or acro in t for t in toks))
EMB, EMB_IDS = None, {}
if os.path.exists(os.path.join(ROOT, 'data', 'meta', 'embeddings.npy')):
    EMB = np.load(os.path.join(ROOT, 'data', 'meta', 'embeddings.npy')).astype(np.float32)
    EMB_IDS = {pid: i for i, pid in enumerate(json.load(open(os.path.join(ROOT, 'data', 'meta', 'embedding_ids.json'))))}
print(f'{len(s2)} Semantic Scholar records, {len(hf)} Hugging Face records, embeddings: {EMB is not None}', flush=True)
NOW = max(m['time'] for m in mentions)
YEAR = 365.25 * 86400

# ---------------------------------------------------------------- stance (v0 rules)
STRONG = re.compile(r"\b(highly recommend\w*|must[- ]read|required reading|seminal|landmark|classic paper|"
                    r"one of (my|the) (favou?rite|best|most important)|(best|greatest|favou?rite) paper|"
                    r"(great|excellent|fantastic|amazing|brilliant|beautiful|wonderful|superb|incredible|outstanding) (paper|read|work|introduction|intro|overview|survey|explanation)|"
                    r"changed (how|the way|my)|(well|really|definitely) worth (reading|a read|the read)|"
                    r"can'?t recommend (it|this) enough|blew my mind|mind[- ]blowing|love (this|that) paper)\b", re.I)
REC = re.compile(r"\b(recommend\w*|worth (a |the )?read\w*|worth reading|check out|you (might|may|will|should) (like|enjoy|read|want)|"
                 r"(good|nice|great|interesting|fascinating|excellent|cool|neat|solid|accessible|readable|clear) (paper|read|intro\w*|overview|survey|summary|explanation|work|resource)|"
                 r"i (really )?(enjoyed|liked|loved)|good starting point|read this|have a look at|take a look at|"
                 r"is a good|very (good|helpful|readable|clear|accessible)|helped me|eye[- ]opening|insightful|elegant|"
                 r"i'?d suggest|i suggest|suggest (this|reading)|(you should|go|definitely|just) read|read the (original )?paper|"
                 r"better understanding|(the )?best (introduction|intro|explanation|overview|treatment|paper)|great resource|"
                 r"worth (your|the) time|classic|landmark|must see|good (one|read)|is great|was great|well written|clearly written)\b", re.I)
CRIT = re.compile(r"\b(flawed|debunk\w*|overhyped|over[- ]hyped|misleading|not convinced|unconvinc\w*|bad paper|p[- ]hack\w*|cherry[- ]pick\w*|"
                  r"doesn'?t (replicate|hold up)|failed to replicate|retracted|poorly written|nonsense|garbage|bullshit|"
                  r"methodolog\w* (problem|issue|flaw)\w*|(i|i'm) (skeptical|sceptical)|snake oil|clickbait|overstat\w*|"
                  r"(paper|study|result|claim)s? (is|are) (wrong|bogus|weak|unconvincing)|not peer[- ]reviewed)\b", re.I)
SELF = re.compile(r"\b((our|my) (recent |new |latest )?(paper|work|preprint|article)|we (wrote|published|released|propose)|i (wrote|co-?authored|published)|(i'?m|i am) (an|the|one of the) (author|co-?author)s?)\b", re.I)
vader = SentimentIntensityAnalyzer()
AUTHOR_STRICT = re.compile(r"\b(i'?m (one of )?the (first |lead |last |senior |co-?)?authors?|i am (one of )?the (first |lead |co-?)?authors?|(author|co-?author|one of the authors) here|this is \w+( \w+)?, (the )?(first |lead |co-?)?author|(\d+(st|nd|rd|th)|first|second|third|last|lead|senior) author (on|of)|i (wrote|co-?wrote|co-?authored|worked on) (this|the|that|our) (paper|work|study)|(our|my) (paper|preprint|work|article|study) (is|was|on|here|shows|found)|disclaimer: i (am|was|co-?authored|wrote|worked)|see (the dynamics discussed in )?my (paper|article))\b", re.I)
SELF_ID = re.compile(r"\b(this is \w+|i'?m \w+,|our (paper|work|team|preprint|results?|model|code)|my (paper|article|preprint|work|thesis)|we (wrote|published|released|trained|built|found|show|benchmark))\b", re.I)


def is_author_comment(m):
    # LLM label + first-person self-identification, or explicit author wording on its own
    t = m.get('text') or ''
    return m['stance'] == 'author' and bool(AUTHOR_STRICT.search(t) or ('llm_quote' in m and SELF_ID.search(t)))


def stance(m, allow_self=True):
    if m['kind'] == 'submission':
        return 'submit'
    t = m.get('text') or ''
    if allow_self and SELF.search(t):       # "our paper" etc.; off for thread comments, where it is often another paper
        return 'author'
    crit, strong, rec = bool(CRIT.search(t)), bool(STRONG.search(t)), bool(REC.search(t))
    v = vader.polarity_scores(t)['compound'] if t else 0.0
    if crit and not strong and v < 0.3:
        return 'critical'
    if strong:
        return 'strong_rec'
    if rec and v > -0.2:
        return 'rec'
    if m.get('substantive') and v >= 0.7 and re.search(r'\b(paper|read|article|work|explain\w*)\b', t, re.I):
        return 'rec'
    if crit:
        return 'critical'
    return 'reference'


for m in mentions:
    m['stance'] = stance(m)
# LLM labels (pipeline/label_llm.py) override the rules where present
llm = {(r['hn_id'], r['paper_id']): r for r in jl(os.path.join(ROOT, 'data', 'labels', 'llm_labels.jsonl'))}
for m in mentions:
    r = llm.get((m['hn_id'], m['paper_id']))
    if r:
        m['stance'] = 'reference' if r['stance'] == 'joke' else r['stance']
        m['llm_quote'], m['why'], m['llm_easy'], m['llm'] = r.get('quote') or '', r.get('why') or '', bool(r.get('easy')), True
LABELER = 'LLM' if llm else 'v0 rules + VADER (LLM pass pending)'
# comments in the HN threads under paper submissions (fetch_algolia.py --threads; no paper link needed).
# They never count as recommendations; they feed "The author on HN" and, when Claude-labeled, "The debate".
THREADS = jl(os.path.join(ROOT, 'data', 'step1', 'thread_comments.jsonl'))
for t in THREADS:
    t['kind'] = 'thread'
    t['stance'] = 'author' if AUTHOR_STRICT.search(t.get('text') or '') else stance(t, allow_self=False)
    r = llm.get((t['hn_id'], t['paper_id']))
    if r:
        t['stance'] = 'reference' if r['stance'] == 'joke' else r['stance']
        t['llm_quote'], t['llm'] = r.get('quote') or '', True
print(f'{len(THREADS)} thread comments ({sum(1 for t in THREADS if t.get("llm"))} Claude-labeled)', flush=True)
print(f'{len(llm)} LLM labels applied', flush=True)
POS = {'strong_rec', 'rec'}


def is_positive(m):
    return m['stance'] in POS or (m['stance'] == 'submit' and (m.get('points') or 0) >= 10)


# ---------------------------------------------------------------- group by paper
by_paper = collections.defaultdict(list)
for m in mentions:
    by_paper[m['paper_id']].append(m)


def ntitle(t):
    t = re.sub(r'\[(pdf|video)\]|\((19|20)\d\d\)', ' ', t or '', flags=re.I)
    return re.sub(r'[^a-z0-9]+', ' ', t.lower()).strip()


# merge venue copies (NeurIPS/PMLR/ACL/OpenReview links) into the arXiv record, or into each other, by title.
# A venue link whose Semantic Scholar record names an arXiv id is canonicalized to that id even when HN never linked
# the arXiv version, as long as we have its arXiv metadata (fetch_arxiv_meta.py fetches it): the paper then gets the
# abstract, authors, categories and embedding.
arxiv_title = {ntitle(md['title']): pid for pid, md in meta.items() if pid in by_paper}
venue_seen, merged_n, canon_n, MERGED = {}, 0, 0, {}
for pid in sorted([p for p in by_paper if not p.startswith('arxiv:')]):
    subs = [m for m in by_paper[pid] if m['kind'] == 'submission' and m.get('story_title')]
    s2a = (s2.get(pid) or {}).get('arxiv_id')
    target = ('arxiv:' + s2a) if s2a and (('arxiv:' + s2a) in by_paper or ('arxiv:' + s2a) in meta) else None
    if target and target not in by_paper:
        canon_n += 1
    nt = ntitle(max(subs, key=lambda m: m.get('points') or 0)['story_title']) if subs else ''
    if not target and len(nt) >= 12:
        target = arxiv_title.get(nt) or venue_seen.get(nt)
    if target and target != pid:
        have = {m['hn_id'] for m in by_paper[target]}
        for m in by_paper.pop(pid):
            if m['hn_id'] not in have:
                m['paper_id'] = target; by_paper[target].append(m)
        MERGED[pid] = target
        if pid in s2 and target not in s2:
            s2[target] = s2[pid]          # keep the citation counts found under the venue id
        merged_n += 1
    elif nt:
        venue_seen.setdefault(nt, pid)
by_thread = collections.defaultdict(list)
for t in THREADS:
    by_thread[MERGED.get(t['paper_id'], t['paper_id'])].append(t)
print(f'merged {merged_n} venue duplicates ({canon_n} canonicalized to an arXiv id HN never linked)', flush=True)

ML_WORDS = re.compile(r"\b(learn\w*|neural|networks?|models?|train\w*|artificial intelligence|machine intelligence|intelligen\w*|agents?|llms?|gpt\w*|"
                      r"transformers?|reinforcement|deep|classif\w*|embedding\w*|language|vision|generative|inference|predict\w*|ai)\b", re.I)


def is_ai(pid):
    if not pid.startswith('arxiv:'):
        return True
    md = meta.get(pid)
    if not (md and set(md['categories']) & AI_CATS):
        return False
    return len(ML_WORDS.findall(md['title'] + ' ' + md['abstract'])) >= 2   # drops e.g. "Magic: The Gathering is Turing Complete"


SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"(\[])')


def best_quote(ms):
    best, best_s = None, -9
    for m in ms:
        if m.get('llm_quote') and m['stance'] in POS and 30 <= len(m['llm_quote']) <= 240:
            sc = 2.0 + (0.5 if m['stance'] == 'strong_rec' else 0) + (0.2 if m.get('substantive') else 0)
            if sc > best_s:
                best_s, best = sc, dict(text=m['llm_quote'], hn_id=m['hn_id'], author=m['author'], month=m['month'], why=m.get('why', ''))
            continue
        if m['stance'] not in POS or not m.get('text'):
            continue
        for s in SENT_SPLIT.split(m['text']):
            s = s.strip()
            if not 40 <= len(s) <= 240 or '[LINK]' in s or s.count('[PAPER]') > 0 or s.startswith(('>', '"')):
                continue
            score = vader.polarity_scores(s)['compound'] + (1.0 if STRONG.search(s) else 0.4 if REC.search(s) else 0)
            score += 0.3 if m['stance'] == 'strong_rec' else 0
            if score > best_s:
                best_s, best = score, dict(text=s, hn_id=m['hn_id'], author=m['author'], month=m['month'])
    return best


BEGINNER = re.compile(r"\b(survey|tutorial|introduction to|an introduction|primer|overview|gentle|explained|for beginners|lecture notes|"
                      r"a guide|guide to|a review|illustrated|hitchhiker|from scratch|crash course|textbook|course notes|book)\b", re.I)
HN_EASY = re.compile(r"\b(good intro\w*|beginner\w*|accessible|easy to (read|follow|understand)|readable|approachable|"
                     r"well[- ]written|clearly written|great (intro\w*|overview|explanation)|good starting point|start with)\b", re.I)
ADV = re.compile(r"\b(theorem|we prove|lemma|upper bounds?|lower bounds?|minimax|regret bounds?|convergence rates?|sample complexity|pac[- ]bayes)\b", re.I)


def difficulty(title, abstract, cats, ms):
    if BEGINNER.search(title or ''):
        return 'beginner'
    n_pos = sum(1 for m in ms if m['stance'] in POS)
    easy = sum(1 for m in ms if m['stance'] in POS and (m['llm_easy'] if 'llm_easy' in m else HN_EASY.search(m.get('text') or '')))
    first = SENT_SPLIT.split(abstract or '')[0] if abstract else ''
    if easy >= 1 and (easy >= 0.1 * n_pos or easy >= 2) or BEGINNER.search(first) or re.search(r'\b(this (survey|tutorial|review|primer)|we (survey|review)|lecture notes|gentle introduction)\b', abstract or '', re.I):
        return 'beginner'
    prim = (cats or [''])[0]
    if prim.startswith('math.') or prim in ('stat.TH', 'cs.IT', 'cs.CC') or len(ADV.findall(abstract or '')) >= 2:
        return 'advanced'
    return 'intermediate'


TOPIC_RE = [(t, re.compile(t['pattern'], re.I)) for t in TOPICS]


def topics_for(title, abstract, cats):
    sc, fallback = [], []
    for t, rx in TOPIC_RE:
        s = 3 * len(rx.findall(title or '')) + min(len(rx.findall(abstract or '')), 4)
        cat = bool(set(cats or []) & set(t['cats']))
        if s:
            sc.append((s + (2 if cat else 0), t['id']))
        elif cat:
            fallback.append((2 if (cats or [''])[0] in t['cats'] else 1, t['id']))
    sc.sort(reverse=True)
    if not sc or sc[0][0] < 2:
        fallback.sort(reverse=True)
        return [fallback[0][1]] if fallback else []
    top = sc[0][0]
    return [tid for s, tid in sc[:3] if s >= max(2, top * 0.4)]


GH = re.compile(r'https?://(?:www\.)?(github\.com|gitlab\.com)/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)', re.I)
GH_SKIP = {'features', 'topics', 'about', 'orgs', 'sponsors', 'marketplace', 'settings', 'pulls', 'issues', 'search'}


def gh_links(text):
    out = []
    for host, owner, repo in GH.findall(text or ''):
        repo = re.sub(r'\.git$', '', repo).rstrip('.')
        if owner.lower() in GH_SKIP or not repo:
            continue
        u = f'https://{host.lower()}/{owner}/{repo}'
        if u not in out:
            out.append(u)
    return out


# raw HN text for GitHub links posted alongside a paper link
raw_text = {}
for name, field in (('comments.jsonl', 'comment_text'), ('stories.jsonl', 'story_text')):
    for h in jl(os.path.join(ROOT, 'hn_algolia', name)):
        if h.get(field) and 'git' in h[field]:
            raw_text[int(h['objectID'])] = html.unescape(h[field])

# curated reading paths (pipeline/make_path.py): their papers are kept even below the recommendation bar
CUR_P = os.path.join(ROOT, 'data', 'curated_paths.json')
CURATED = json.load(open(CUR_P, encoding='utf-8')) if os.path.exists(CUR_P) else []
CUR_IDS = {s['id'] for c in CURATED for s in c['steps']}

# channel videos per paper (data/paper_videos.json: {"arxiv:1706.03762": "https://www.youtube.com/watch?v=..." or 11-char id})
YT_RE = re.compile(r'(?:youtu\.be/|youtube\.com/(?:watch\?[^"\s]*v=|shorts/|embed/))([A-Za-z0-9_-]{11})|^([A-Za-z0-9_-]{11})$')
PAPER_YT = {}
_pvp = os.path.join(ROOT, 'data', 'paper_videos.json')
if os.path.exists(_pvp):
    with open(_pvp, encoding='utf-8') as _f:
        for _k, _v in json.load(_f).items():
            _pid = _k.strip() if ':' in _k else f'arxiv:{_k.strip()}'
            _m = YT_RE.search(str(_v or '').strip())
            if _m:
                PAPER_YT[_pid] = _m.group(1) or _m.group(2)

# ---------------------------------------------------------------- per-paper records
papers = []
for pid, ms in by_paper.items():
    if not is_ai(pid) and pid not in CUR_IDS:
        continue
    pos_users = {m['author'] for m in ms if is_positive(m)}
    has_author = any(is_author_comment(m) for m in ms)
    n_people = len({m['author'] for m in ms if m['stance'] != 'spam'})
    below = len(pos_users) < a.min_positive and not (has_author and len(pos_users) >= 1 and n_people >= 2)
    if below and pid not in CUR_IDS:
        continue
    md = meta.get(pid, {})
    subs = [m for m in ms if m['kind'] == 'submission']
    title = md.get('title') or (max(subs, key=lambda m: m.get('points') or 0)['story_title'] if subs else None)
    if not title:
        continue
    abstract = md.get('abstract') or ''
    cats = ([md['primary_category']] + [c for c in md.get('categories', []) if c != md.get('primary_category')]) if md else [pid.split(':')[0]]
    pub = md.get('published')
    year = int(pub[:4]) if pub else int(min(m['month'] for m in ms)[:4])
    users = {m['author'] for m in ms if m['stance'] not in ('author', 'spam')}
    strong = {m['author'] for m in ms if m['stance'] == 'strong_rec'}
    critical = {m['author'] for m in ms if m['stance'] == 'critical'}
    pos_months = {m['month'] for m in ms if is_positive(m)}
    max_pts = max([m.get('points') or 0 for m in subs] or [0])
    last = max(m['time'] for m in ms)
    recent = [m for m in ms if m['time'] >= NOW - YEAR]
    rec_pos = {m['author'] for m in recent if is_positive(m)}
    rec_pts = max([m.get('points') or 0 for m in recent if m['kind'] == 'submission'] or [0])

    base = (3.0 * math.log1p(len(pos_users)) + 1.5 * math.log1p(len(users)) + 1.0 * math.log1p(len(pos_months))
            + 0.6 * math.log1p(max_pts) + 0.5 * math.log1p(len(strong)) - 0.5 * math.log1p(len(critical))
            + 0.35 * math.log1p((s2.get(pid) or {}).get('citations') or 0))
    recency = 1.2 * math.exp(-(NOW - last) / YEAR / 2)
    trend = (3.0 * math.log1p(len(rec_pos)) + 1.0 * math.log1p(len({m['author'] for m in recent})) + 0.6 * math.log1p(rec_pts)) if recent else 0

    code = gh_links((md.get('comment') or '') + ' ' + abstract)
    code_src = 'paper' if code else None
    if not code:
        for m in ms:
            if is_author_comment(m):
                g = gh_links(raw_text.get(m['hn_id'], ''))
                if g:
                    code, code_src = g[:1], 'author'; break
    aid0 = pid.split(':', 1)[1]
    if not code and aid0 in HF_CODE:
        u = HF_CODE[aid0][0]
        if aid0 in HF_HUMAN or repo_matches(md.get('title') or '', u):
            code, code_src = [u], 'hf'
    if not code:
        cnt = collections.Counter()
        for m in ms:
            for u in gh_links(raw_text.get(m['hn_id'], '')):
                cnt[u] += 1
        # an HN-posted repo counts only if its name matches the paper (title word or arXiv id)
        tw = {w for w in re.findall(r'[a-z0-9]{4,}', (md.get('title') or '').lower())} - {'with', 'from', 'that', 'this', 'learning', 'neural', 'model', 'models', 'language', 'large', 'networks', 'deep', 'towards', 'using', 'via'}
        aid = pid.split(':', 1)[1].replace('.', '')
        ok = [u for u, _ in cnt.most_common() if aid in u.replace('.', '') or (tw & set(re.findall(r'[a-z0-9]{4,}', u.lower().split('/', 3)[-1])))]
        if ok:
            code, code_src = ok[:1], 'hn'

    sents = SENT_SPLIT.split(abstract) if abstract else []
    tldr = ''
    for s in sents:
        if len(tldr) + len(s) > 360 and tldr:
            break
        tldr = (tldr + ' ' + s).strip()

    tps = topics_for(title, abstract, cats)
    lvl = difficulty(title, abstract, cats, ms)
    stance_counts = collections.Counter(m['stance'] for m in ms)

    # mentions shown on the paper page: positives first, one per person, substantive first
    shown, seen = [], set()
    order = sorted([m for m in ms if m['kind'] != 'submission' and m.get('text')],
                   key=lambda m: (m['stance'] not in POS, m['stance'] == 'author', not m.get('substantive'), -m['time']))
    for m in order:
        if m['author'] in seen or len(shown) >= 16:
            continue
        seen.add(m['author'])
        txt = m['text']
        shown.append(dict(id=m['hn_id'], a=m['author'], mo=m['month'], st=m['stance'],
                          t=txt[:700] + ('…' if len(txt) > 700 else ''), sid=m.get('story_id'),
                          stt=(m.get('story_title') or '')[:140]))
    stories = sorted([dict(id=m['hn_id'], t=m.get('story_title'), pts=m.get('points') or 0, c=m.get('num_comments') or 0,
                           mo=m['month']) for m in subs], key=lambda s: -s['pts'])[:6]

    # --- HN over time: per-year counts, hype vs lasting, then vs now
    yr = collections.defaultdict(lambda: [0, 0, 0])
    for m in ms:
        y = int(m['month'][:4]); yr[y][2] += 1
        if is_positive(m): yr[y][0] += 1
        if m['stance'] == 'critical': yr[y][1] += 1
    timeline = [[y] + yr[y] for y in range(min(yr), max(yr) + 1)]
    pos_years = sorted({int(mo[:4]) for mo in pos_months})
    span = (pos_years[-1] - pos_years[0]) if pos_years else 0
    mcount = collections.Counter(m['month'] for m in ms)
    peak_share = max(mcount.values()) / len(ms)
    # hype vs lasting: HN timing plus academic uptake (citations per year since publication)
    cites_ = (s2.get(pid) or {}).get('citations')
    cpy = (cites_ / max(1.0, datetime.fromtimestamp(NOW, timezone.utc).year - year + 0.5)) if cites_ is not None else None
    hn_lasting = len(pos_months) >= 4 and span >= 3
    cited_lasting = len(pos_months) >= 3 and span >= 2 and cpy is not None and cpy >= 100
    hn_spike = peak_share >= 0.6 and span <= 1 and len(ms) >= 6
    lasting = 'lasting' if (hn_lasting or cited_lasting) else ('spike' if hn_spike and (cpy is None or cpy < 50) else '')

    def snap(m):
        q = m.get('llm_quote') or next((x for x in SENT_SPLIT.split(m.get('text') or '') if 30 <= len(x) <= 240 and '[LINK]' not in x), '')
        return dict(id=m['hn_id'], a=m['author'], mo=m['month'], st=m['stance'], q=q.strip()) if q else None
    t0 = min(m['time'] for m in ms)
    opinions = sorted([m for m in ms if m['kind'] != 'submission' and m.get('text') and m['stance'] in ('strong_rec', 'rec', 'critical')],
                      key=lambda m: m['time'])
    then = next((x for x in (snap(m) for m in opinions if m['time'] <= t0 + YEAR) if x), None)
    now_ = next((x for x in (snap(m) for m in reversed(opinions) if m['time'] >= t0 + 2 * YEAR) if x), None)

    # --- the author on HN, and the pushback
    th = by_thread.get(pid, [])
    sub_title = (max(subs, key=lambda m: m.get('points') or 0).get('story_title') or '') if subs else ''
    au_seen, au_rows = set(), []
    for m in sorted([m for m in ms + th if m['stance'] == 'author' and m.get('text') and is_author_comment(m)], key=lambda m: m['time']):
        if m['hn_id'] not in au_seen:
            au_seen.add(m['hn_id']); au_rows.append(m)
    authors_hn = [dict(id=m['hn_id'], a=m['author'], mo=m['month'], t=m['text'][:450] + ('…' if len(m['text']) > 450 else ''),
                       **({'th': 1} if m['kind'] == 'thread' else {})) for m in au_rows][:3]
    pushback = [dict(id=m['hn_id'], a=m['author'], mo=m['month'], q=(m.get('llm_quote') or ''), t=m['text'][:500] + ('…' if len(m['text']) > 500 else ''),
                     stt=(m.get('story_title') or sub_title)[:140])
                for m in sorted([m for m in ms if m['stance'] == 'critical' and m.get('text')]
                                + [t for t in th if t.get('llm') and t['stance'] == 'critical' and t.get('substantive')],
                                key=lambda m: (not m.get('substantive'), -len(m.get('llm_quote') or ''), -m['time']))][:4]
    shown = [x for x in shown if x['st'] not in ('critical', 'author', 'spam')][:12]

    src_url = next((m['raw_url'] for m in ms if m.get('raw_url', '').startswith('http')), None)
    papers.append(dict(
        timeline=timeline, lasting=lasting, peak_share=round(peak_share, 2), span=span, then=then, now=now_,
        cites_per_year=round(cpy) if cpy is not None else None,
        authors_hn=authors_hn, pushback=pushback,
        id=pid, url=None if pid.startswith('arxiv:') else src_url, title=title, year=year, published=(pub or '')[:10], authors=md.get('authors', [])[:12],
        n_authors=len(md.get('authors', [])), cats=cats[:4], abstract=abstract, tldr=tldr, topics=tps, level=lvl,
        users=len(users), pos=len(pos_users), strong=len(strong), critical=len(critical), months=len(pos_months),
        points=max_pts, first=min(m['month'] for m in ms), last=datetime.fromtimestamp(last, timezone.utc).strftime('%Y-%m'),
        score=round(base + recency, 4), base=round(base, 4), trend=round(trend, 4), recent_pos=len(rec_pos),
        code=code[:2], code_src=code_src, quote=best_quote(ms), stances=dict(stance_counts),
        cites=(s2.get(pid) or {}).get('citations'), infl=(s2.get(pid) or {}).get('influential'),
        venue=(s2.get(pid) or {}).get('venue') or '', s2_url=(s2.get(pid) or {}).get('url'),
        hf=dict({k: (hf.get(pid) or {}).get(k, [])[:5] for k in ('models', 'datasets', 'spaces')}, page=bool(hf[pid].get('hf_paper'))) if pid in hf else None,
        mentions=shown, stories=stories, _users=pos_users | users, n_hn=len(ms), xp=below or not is_ai(pid),
    ))
print(f'{sum(not p["xp"] for p in papers)} AI papers with >= {a.min_positive} positive users', flush=True)

# editor's picks: curated papers HN never linked (built from arXiv metadata alone)
_have = {p['id'] for p in papers}
for pid in sorted(CUR_IDS - _have):
    md = meta.get(pid)
    if not md:
        print(f'  curated paper {pid} has no arXiv metadata: skipped (run make_path.py online)', flush=True)
        continue
    abstract = md.get('abstract') or ''
    cats = [md['primary_category']] + [c for c in md.get('categories', []) if c != md.get('primary_category')]
    tldr = ''
    for s_ in (SENT_SPLIT.split(abstract) if abstract else []):
        if len(tldr) + len(s_) > 360 and tldr:
            break
        tldr = (tldr + ' ' + s_).strip()
    code = gh_links((md.get('comment') or '') + ' ' + abstract)
    year = int(md['published'][:4])
    papers.append(dict(
        timeline=[], lasting='', peak_share=0, span=0, then=None, now=None, cites_per_year=None, authors_hn=[], pushback=[],
        id=pid, url=None, title=md['title'], year=year, published=md['published'][:10], authors=md.get('authors', [])[:12],
        n_authors=len(md.get('authors', [])), cats=cats[:4], abstract=abstract, tldr=tldr, topics=topics_for(md['title'], abstract, cats),
        level=difficulty(md['title'], abstract, cats, []), users=0, pos=0, strong=0, critical=0, months=0, points=0, first='', last='',
        score=0, base=0, trend=0, recent_pos=0, code=code[:2], code_src='paper' if code else None, quote=None, stances={},
        cites=(s2.get(pid) or {}).get('citations'), infl=(s2.get(pid) or {}).get('influential'),
        venue=(s2.get(pid) or {}).get('venue') or '', s2_url=(s2.get(pid) or {}).get('url'),
        hf=None, mentions=[], stories=[], _users=set(), n_hn=0, xp=True, pick=True))
print(f'{sum(p["xp"] for p in papers)} extra papers kept for curated paths', flush=True)

# ---------------------------------------------------------------- topics: keyword rules, checked against embeddings
# Each topic gets a centroid from the papers whose first rule-based topic it is. A paper with no rule topic gets
# its nearest centroid; a paper whose nearest centroid is clearly closer than its rule topic gets that topic first.
TOPIC_CHANGES = collections.Counter()
if EMB is not None:
    _rows = [EMB_IDS.get(p['id']) for p in papers]
    _tids = [t['id'] for t in TOPICS]
    _cent, _keep = [], []
    for tid in _tids:
        mem = [r for p, r in zip(papers, _rows) if r is not None and not p['xp'] and p['topics'] and p['topics'][0] == tid]
        if len(mem) >= 5:
            c = EMB[mem].mean(0); _cent.append(c / np.linalg.norm(c)); _keep.append(tid)
    if _cent:
        C_ = np.array(_cent, np.float32)
        for p, r in zip(papers, _rows):
            if r is None:
                continue
            sims = C_ @ EMB[r]
            best = int(np.argmax(sims)); bt = _keep[best]
            if not p['topics']:
                p['topics'] = [bt]; TOPIC_CHANGES['filled'] += 1
            elif bt not in p['topics'] and p['topics'][0] in _keep and sims[best] - sims[_keep.index(p['topics'][0])] > 1.5 * a.topic_margin:
                p['topics'] = [bt] + p['topics'][:2]; TOPIC_CHANGES['added first'] += 1
            elif bt in p['topics'] and p['topics'][0] != bt and p['topics'][0] in _keep and sims[best] - sims[_keep.index(p['topics'][0])] > a.topic_margin:
                p['topics'] = [bt] + [t for t in p['topics'] if t != bt]; TOPIC_CHANGES['reordered'] += 1
print(f'topics: keyword rules + embedding check {dict(TOPIC_CHANGES)}', flush=True)

# ---------------------------------------------------------------- related + read-first
idx = {p['id']: i for i, p in enumerate(papers)}
docs = [f"{p['title']} {p['title']} {p['abstract']} " + ' '.join(t.replace('-', ' ') for t in p['topics']) for p in papers]
X = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True).fit_transform(docs)
S = X @ X.T
S = (S.toarray() if hasattr(S, 'toarray') else np.asarray(S)).astype(np.float32)
if EMB is not None:
    rows_e = [EMB_IDS.get(p['id']) for p in papers]
    have = np.array([r is not None for r in rows_e])
    Ep = np.zeros((len(papers), EMB.shape[1]), np.float32)
    Ep[have] = EMB[[r for r in rows_e if r is not None]]
    Se = np.clip((Ep @ Ep.T - 0.6) / 0.35, 0, 1)          # bge cosine ~0.6 (unrelated) .. 0.95 (near-duplicate)
    both = np.outer(have, have)
    S = np.where(both, 0.35 * S + 0.65 * Se, S).astype(np.float32)
np.fill_diagonal(S, 0)

user_papers = collections.defaultdict(set)
for i, p in enumerate(papers):
    for u in p['_users']:
        user_papers[u].add(i)
shared = collections.defaultdict(collections.Counter)
for u, ps in user_papers.items():
    if len(ps) > 300:          # very prolific linkers carry little signal
        continue
    ps = list(ps)
    for i in ps:
        for j in ps:
            if i != j:
                shared[i][j] += 1

for i, p in enumerate(papers):
    cand = set(np.argsort(-S[i])[:30].tolist()) | {j for j, _ in shared[i].most_common(30)}
    scored = []
    for j in cand:
        if papers[j]['xp'] and not p['xp']:
            continue
        sh, sim = shared[i][j], float(S[i, j])
        if sim < 0.08 and sh < 2:
            continue
        scored.append((0.6 * sim + 0.4 * min(sh, 5) / 5, j, sh, sim))
    scored.sort(reverse=True)
    p['related'] = [dict(id=papers[j]['id'], shared=sh, sim=round(sim, 3)) for _, j, sh, sim in scored[:8]]
    pre = [(S[i, j] * math.log1p(papers[j]['users']), j) for j in np.argsort(-S[i])[:40]
           if S[i, j] >= 0.12 and papers[j]['year'] <= p['year'] - 1 and papers[j]['users'] >= max(5, p['users'] // 2)]
    pre.sort(reverse=True)
    p['prereqs'] = [papers[j]['id'] for _, j in pre[:3]]

# ---------------------------------------------------------------- tabs + output
papers.sort(key=lambda p: -p['score'])
cur_year = datetime.fromtimestamp(NOW, timezone.utc).year
for p in papers:
    p['classic'] = p['year'] <= cur_year - 3 and p['months'] >= 3
    p['trending'] = p['recent_pos'] >= 1 and p['trend'] > 0

# ---------------------------------------------------------------- reading paths (ordered oldest -> newest)
TRACKS = [
    ('transformers-from-zero', 'Transformers from zero', 'From attention and word vectors to BERT, GPT and the models HN argues about today.',
     ['architectures', 'large-language-models', 'long-context-attention', 'classic-nlp', 'self-supervised']),
    ('scaling-laws', 'Scaling laws and emergence', 'Why bigger models kept getting better, and the fights over "emergent" abilities.',
     ['scaling-laws', 'large-language-models', 'data-datasets']),
    ('agents', 'LLM agents', 'Tool use, planning and agents that act, plus the papers HN uses to check the hype.',
     ['agents-tool-use', 'planning-search', 'prompting-reasoning', 'code-generation']),
    ('alignment', 'Fine-tuning, RLHF and alignment', 'How raw language models become assistants, and what can go wrong.',
     ['alignment-finetuning', 'ai-safety', 'hallucination']),
    ('interpretability', 'Looking inside the model', 'Probing, circuits and features: what HN reads to understand what networks learn.',
     ['interpretability', 'dl-theory-optimization']),
    ('generative', 'Generative images and audio', 'GANs, VAEs, diffusion and the audio models HN keeps linking.',
     ['diffusion', 'gans', 'vae-flows', 'image-video-generation', 'speech-audio']),
    ('efficiency', 'Making models small and fast', 'Quantization, distillation, mixture of experts and the systems work behind cheap inference.',
     ['efficient-inference', 'mixture-of-experts', 'ml-systems']),
    ('reinforcement-learning', 'Reinforcement learning and games', 'From Atari and AlphaGo to RL for reasoning models.',
     ['reinforcement-learning', 'games', 'robotics']),
]
paths = []
for tid, name, blurb, tps in TRACKS:
    cand = [p for p in papers if not p['xp'] and p['topics'] and (p['topics'][0] in tps[:2] or (p['topics'][0] in tps and tps[0] in p['topics']))]
    cand.sort(key=lambda p: -(p['base'] + (2.0 if p['classic'] else 0) + (1.5 if p['level'] == 'beginner' else 0)
                              + (2.0 if p['topics'][0] == tps[0] else 0)))
    steps = sorted(cand[:9], key=lambda p: (p['year'], p['published']))
    if len(steps) >= 4:
        paths.append(dict(id=tid, name=name, blurb=blurb, topics=tps, steps=[p['id'] for p in steps]))
        for i, p in enumerate(steps):
            p.setdefault('paths', []).append(dict(id=tid, name=name, step=i + 1, of=len(steps)))
_pmap = {p['id']: p for p in papers}
for c in CURATED:
    steps = [s for s in c['steps'] if s['id'] in _pmap]
    if not steps:
        continue
    info = {s['id']: dict(hn=len(by_paper.get(s['id'], [])), people=len({m['author'] for m in by_paper.get(s['id'], [])}),
                          **({k: s[k] for k in ('pick', 'why', 'stage', 'core') if s.get(k) is not None and s.get(k) is not False})) for s in steps}
    paths.append(dict(id=c['id'], name=c['name'], blurb=c['blurb'], topics=[], curated=True, updated=c.get('updated', ''),
                      steps=[s['id'] for s in steps], info=info, **({'stages': c['stages']} if c.get('stages') else {})))
    for i, s in enumerate(steps):
        _pmap[s['id']].setdefault('paths', []).append(dict(id=c['id'], name=c['name'], step=i + 1, of=len(steps)))
json.dump(paths, open(os.path.join(OUT, 'paths.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
print(f'{len(paths)} reading paths ({sum(1 for x in paths if x.get("curated"))} curated)', flush=True)

# ---------------------------------------------------------------- HackerTube videos (optional: data/hackertube/ from hackertube.org/data)
HT = os.path.join(ROOT, 'data', 'hackertube')
if not os.path.exists(os.path.join(HT, 'videos.jsonl')) and os.path.exists(os.path.join(HT, 'hackertube-data', 'videos.jsonl')):
    HT = os.path.join(HT, 'hackertube-data')   # the zip unpacks into a subfolder
n_vid = 0
if EMB is not None and os.path.exists(os.path.join(HT, 'videos.jsonl')) and os.path.exists(os.path.join(HT, 'embeddings.npy')):
    vids = {v['vid']: v for v in jl(os.path.join(HT, 'videos.jsonl'))}
    VE = np.load(os.path.join(HT, 'embeddings.npy')).astype(np.float32)
    v_ids = json.load(open(os.path.join(HT, 'embedding_ids.json')))
    VE /= np.linalg.norm(VE, axis=1, keepdims=True) + 1e-9
    STOP = set('the and for with from that this are via using towards into your what how why all you need large model models learning neural networks deep'.split())
    # videos linked in the same HN comment as a paper (HackerTube mentions carry the HN comment id)
    vid_by_comment = collections.defaultdict(set)
    for hm in jl(os.path.join(HT, 'mentions.jsonl')):
        if hm.get('hn_comment_id') and hm.get('vid'):
            vid_by_comment[int(hm['hn_comment_id'])].add(hm['vid'])
    pos_of = {v: k for k, v in enumerate(v_ids)}
    vdf = collections.Counter(w for v in vids.values() for w in set(re.findall(r'[a-z0-9]{4,}', ntitle(v.get('title', '')))))

    def about(nt, keys, vt):
        # exact paper title in the video title, or 2+ words that are rare among HackerTube titles (df <= 6)
        rare = [w for w in keys if vdf.get(w, 0) <= 6]
        return (len(nt) >= 12 and nt in vt) or sum(w in vt.split() for w in rare) >= 2
    for p in papers:
        j = EMB_IDS.get(p['id'])
        if j is None:
            continue
        sims = VE @ EMB[j]
        keys = {w for w in re.findall(r'[a-z0-9]{4,}', p['title'].lower()) if w not in STOP}
        nt = ntitle(p['title'])
        found = {}
        for m in by_paper.get(p['id'], []):
            for vid in vid_by_comment.get(m['hn_id'], ()):
                if vid in vids and vid in pos_of:
                    found.setdefault(vid, ('together', float(sims[pos_of[vid]])))
        for k in np.argsort(-sims)[:15]:
            v = vids.get(v_ids[k]); sim = float(sims[k])
            if not v or v['vid'] in found:
                continue
            vt = ntitle(v.get('title', ''))
            hit = sum(w in vt for w in keys)
            if about(nt, keys, vt):
                found[v['vid']] = ('about', sim)
            elif sim >= 0.80 and keys and sum(w in (vt + ' ' + v.get('summary', '').lower()) for w in keys) >= max(1, min(2, len(keys) // 2)):
                found[v['vid']] = ('related', sim)
        for vid, (kind, sim) in list(found.items()):   # a co-linked video whose title matches is about the paper
            vt = ntitle(vids[vid].get('title', ''))
            if kind == 'together' and about(nt, keys, vt):
                found[vid] = ('about', sim)
        order = {'about': 0, 'together': 1, 'related': 2}
        picks = sorted(found.items(), key=lambda kv: (order[kv[1][0]], -kv[1][1]))[:3]
        if picks:
            p['videos'] = [dict(vid=vid, kind=kind, title=vids[vid].get('title'), channel=vids[vid].get('channel'),
                                why=vids[vid].get('why_hn', ''), sim=round(sim, 3)) for vid, (kind, sim) in picks]
            n_vid += 1
print(f'{n_vid} papers matched to HackerTube videos', flush=True)


def shard(pid):
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % N_SHARDS


light_keys = ['id', 'title', 'year', 'topics', 'level', 'users', 'pos', 'strong', 'months', 'points', 'score', 'base', 'trend', 'cites',
              'classic', 'trending', 'first', 'last']
index = []
for p in papers:
    if p['id'] in PAPER_YT:
        p['yt'] = PAPER_YT[p['id']]
    r = {k: p[k] for k in light_keys}
    r['au'] = (p['authors'][0] + (' et al.' if p['n_authors'] > 1 else '')) if p['authors'] else ''
    r['code'] = bool(p['code'])
    if p.get('yt'):
        r['yt'] = p['yt']
    r['q'] = p['quote']['text'] if p['quote'] else ''
    if p['quote'] and p['quote'].get('why'):
        r['why'] = p['quote']['why']
    r['tl'] = p['tldr'][:220]
    r['cat'] = p['cats'][0] if p['cats'] else ''
    r['sh'] = shard(p['id'])
    r['slug'] = re.sub(r'[^A-Za-z0-9]+', '-', p['id']).strip('-')
    r['lasting'] = p['lasting']
    r['auth'] = bool(p['authors_hn'])
    r['crit'] = len(p['pushback'])
    r['deb'] = (len(p['pushback']) >= 2 and p['pos'] >= 2) or (len(p['pushback']) >= 1 and p['pos'] >= 3)
    if p.get('url'):
        r['url'] = p['url']
    if p['xp']:
        r['xp'] = 1          # path-only: below the bar, kept out of the feeds, search and For You
        r['hn'] = p['n_hn']
    if p.get('pick'):
        r['pick'] = 1
    index.append(r)
assert len({r['slug'] for r in index}) == len(index), 'slug collision'

shards = collections.defaultdict(dict)
for p in papers:
    d = {k: v for k, v in p.items() if not k.startswith('_')}
    shards[shard(p['id'])][p['id']] = d
for s in range(N_SHARDS):
    json.dump(shards.get(s, {}), open(os.path.join(OUT, 'p', f'{s}.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))

tcount = collections.Counter(t for p in papers for t in p['topics'])
taxo = dict(version='v1-rules+embeddings', areas=[dict(id=aid, name=an, topics=[dict(id=t[0], name=t[1], n=tcount.get(t[0], 0)) for t in ts])
                                       for aid, an, ts in AREAS])
all_stance = collections.Counter(m['stance'] for m in mentions)
MAINP = [p for p in papers if not p["xp"]]
stats = dict(built=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
             data_through=datetime.fromtimestamp(NOW, timezone.utc).strftime('%Y-%m-%d'),
             papers=len(MAINP), mentions_total=len(mentions), stance_counts=dict(all_stance),
             hn_users=len({m['author'] for m in mentions}), classic=sum(p['classic'] for p in MAINP),
             trending=sum(p['trending'] for p in MAINP), with_code=sum(bool(p['code']) for p in MAINP),
             beginner=sum(p['level'] == 'beginner' for p in MAINP), labeler=LABELER,
             with_citations=sum(p['cites'] is not None for p in MAINP), with_hf=sum(bool(p['hf'] and any(p['hf'].values())) for p in MAINP),
             embeddings='bge-small-en-v1.5' if EMB is not None else 'tf-idf only', paths=len(paths), with_videos=n_vid,
             lasting=sum(p['lasting'] == 'lasting' for p in MAINP), spike=sum(p['lasting'] == 'spike' for p in MAINP),
             with_author=sum(bool(p['authors_hn']) for p in MAINP), debated=sum((len(p['pushback']) >= 2 and p['pos'] >= 2) or (len(p['pushback']) >= 1 and p['pos'] >= 3) for p in MAINP),
             code_by_source=dict(collections.Counter(p['code_src'] for p in MAINP if p['code'])), path_only=sum(p['xp'] for p in papers), llm_labeled=sum(1 for p in MAINP for m in by_paper.get(p['id'], []) if m['kind'] != 'submission' and m.get('text') and m.get('llm')),
             comments_with_text=sum(1 for p in MAINP for m in by_paper.get(p['id'], []) if m['kind'] != 'submission' and m.get('text')))
json.dump(index, open(os.path.join(OUT, 'index.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
# words.json: word -> index rows whose title or full abstract uses it. The page loads it only when a search finds no
# title/TL;DR/quote match, so concepts named in abstracts still anchor the meaning-based search.
_W_STOP = set('this that with from have been were their which these those into than then also such more most other some only over '
              'each both they them very what when where while about across after among based between using used uses show shows '
              'shown paper papers propose proposed present work results result approach method methods model models task tasks'.split())
_pmap2 = {p['id']: p for p in papers}
postings = collections.defaultdict(list)
for i, r in enumerate(index):
    p = _pmap2[r['id']]
    ws = set()
    for w in re.findall(r'[a-z][a-z0-9-]{3,}', (p['title'] + ' ' + (p['abstract'] or '')).lower()):
        w = w.strip('-')
        if len(w) > 4 and w.endswith('s') and not w.endswith('ss'):
            w = w[:-1]
        if len(w) >= 4 and w not in _W_STOP:
            ws.add(w)
    for w in ws:
        postings[w].append(i)
words = {w: rows for w, rows in postings.items() if 2 <= len(rows) <= max(40, len(index) // 12)}
json.dump(words, open(os.path.join(OUT, 'words.json'), 'w', encoding='utf-8'), separators=(',', ':'))
print(f'words.json: {len(words)} words', flush=True)
# int8 embeddings for the site papers, in index order (for For You + meaning-based search in the browser)
if EMB is not None:
    import base64
    V = np.zeros((len(index), EMB.shape[1]), np.float32)
    for i, r in enumerate(index):
        j = EMB_IDS.get(r['id'])
        if j is not None:
            V[i] = EMB[j]
    Q = np.clip(np.round(V * 127), -127, 127).astype(np.int8)
    json.dump(dict(model='bge-small-en-v1.5', dim=int(Q.shape[1]), n=int(Q.shape[0]), scale=127,
                   b64=base64.b64encode(Q.tobytes()).decode()), open(os.path.join(OUT, 'emb.json'), 'w'))
json.dump(taxo, open(os.path.join(OUT, 'taxonomy.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump(stats, open(os.path.join(OUT, 'stats.json'), 'w', encoding='utf-8'), indent=1)
print(json.dumps(stats, indent=1))
