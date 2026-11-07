"""Find AI-paper links in HN text and normalize them to canonical paper IDs.

Canonical ID formats
  arxiv:1706.03762          new-style arXiv (version stripped)
  arxiv:hep-th/9901001      old-style arXiv (subject class like ".AI" stripped)
  openreview:<forum id>
  acl:2020.acl-main.463 | acl:P19-1001
  neurips:<32-hex hash>     modern papers.nips.cc / proceedings.neurips.cc URLs
  neurips:n<number>         legacy papers.nips.cc/paper/<n>-slug URLs
  pmlr:v119/chen20j

Non-arXiv IDs are mapped to arXiv IDs later (step 3, via Semantic Scholar
externalIds) when the paper has an arXiv version.
"""
import html
import re
from urllib.parse import urlsplit, parse_qs, unquote

DOMAINS = ("arxiv.org", "ar5iv.org", "openreview.net", "aclanthology.org",
           "aclweb.org", "papers.nips.cc", "proceedings.neurips.cc",
           "proceedings.mlr.press", "doi.org/10.48550")

# Bare or href'd URLs inside HN comment HTML (after unescaping).
URL_RE = re.compile(r'https?://[^\s"\'<>]+', re.I)
# Scheme-less mentions such as "arxiv.org/abs/2301.00001" in plain text.
BARE_RE = re.compile(r'(?<![\w/.])((?:www\.|export\.)?(?:arxiv\.org|openreview\.net|aclanthology\.org|proceedings\.mlr\.press|papers\.nips\.cc|proceedings\.neurips\.cc)/[^\s"\'<>]+)', re.I)
# Plain-text "arXiv:2301.00001" references.
ARXIV_TAG_RE = re.compile(r'\barxiv\s*:\s*(\d{4}\.\d{4,5})(?:v\d+)?\b', re.I)

NEW_ARXIV = r'(\d{4}\.\d{4,5})(?:v\d+)?'
OLD_ARXIV = r'([a-z][a-z\-]+)(?:\.[A-Za-z]{2})?/(\d{7})(?:v\d+)?'
ARXIV_PATH_NEW = re.compile(r'^/(?:abs|pdf|html|format|ps|src|v\d*|papers?)/' + NEW_ARXIV + r'(?:\.pdf)?/?', re.I)
ARXIV_PATH_OLD = re.compile(r'^/(?:abs|pdf|html|format|ps|src)/' + OLD_ARXIV + r'(?:\.pdf)?/?', re.I)
ARXIV_ANY_NEW = re.compile(r'(?<![\d.])' + NEW_ARXIV + r'(?![\d])')

ACL_ID = re.compile(r'(\d{4}\.[a-z0-9]+(?:-[a-z0-9]+)*\.\d+|[A-Z]\d{2}-\d{4})', re.I)
NIPS_HASH = re.compile(r'/(?:paper_files/)?paper/\d{4}/(?:hash|file)/([0-9a-f]{32})-', re.I)
NIPS_LEGACY = re.compile(r'^/paper/(\d{3,5})-[^/]*', re.I)
PMLR = re.compile(r'^/(v\d+|r\d+)/([A-Za-z0-9_\-]+?)(?:\.html|\.pdf|/[^/]*)?/?$', re.I)

TRAIL = '.,;:!?)]}>\'"*'

# arXiv archives that used old-style IDs (archive/YYMMNNN); lookup ignores hyphens so
# typos like "quantph" or "condmat" map to the real name
OLD_ARCHIVES = ['acc-phys', 'adap-org', 'alg-geom', 'ao-sci', 'astro-ph', 'atom-ph', 'bayes-an', 'chao-dyn',
                'chem-ph', 'cmp-lg', 'comp-gas', 'cond-mat', 'cs', 'dg-ga', 'funct-an', 'gr-qc', 'hep-ex',
                'hep-lat', 'hep-ph', 'hep-th', 'math', 'math-ph', 'mtrl-th', 'nlin', 'nucl-ex', 'nucl-th',
                'patt-sol', 'physics', 'plasm-ph', 'q-alg', 'q-bio', 'quant-ph', 'solv-int', 'supr-con']
_ARCH = {a.replace('-', ''): a for a in OLD_ARCHIVES}


def clean_url(u: str) -> str:
    u = html.unescape(u).strip()
    while u and u[-1] in TRAIL:
        # keep a ')' only if it is balanced inside the URL
        if u[-1] == ')' and u.count('(') >= u.count(')'):
            break
        u = u[:-1]
    return u


def normalize(url: str):
    """Return (paper_id, source) for a paper URL, or None for non-paper pages."""
    try:
        parts = urlsplit(url if '://' in url else 'https://' + url)
    except ValueError:
        return None
    host = (parts.hostname or '').lower()
    if host.startswith('www.'):
        host = host[4:]
    path = unquote(parts.path or '')
    q = parse_qs(parts.query)

    if host.endswith('arxiv.org') or host == 'ar5iv.org':
        m = ARXIV_PATH_NEW.match(path)
        if m:
            return 'arxiv:' + m.group(1), 'arxiv'
        m = ARXIV_PATH_OLD.match(path)
        if m:
            arch = _ARCH.get(m.group(1).lower().replace('-', ''))
            return (f'arxiv:{arch}/{m.group(2)}', 'arxiv') if arch else None
        if host.startswith('ar5iv') or host.startswith('alphaxiv'):
            m = ARXIV_ANY_NEW.search(path)
            if m:
                return 'arxiv:' + m.group(1), 'arxiv'
        return None

    if host == 'doi.org' and path.lower().startswith('/10.48550/arxiv.'):
        m = re.search(r'arxiv\.(\d{4}\.\d{4,5})', path, re.I)
        return ('arxiv:' + m.group(1), 'arxiv') if m else None

    if host == 'openreview.net':
        if re.match(r'^/(forum|pdf|attachment|revisions)/?$', path):
            ids = q.get('id')
            if ids and re.fullmatch(r'[A-Za-z0-9_\-]{5,}', ids[0]):
                return 'openreview:' + ids[0], 'openreview'
        return None

    if host in ('aclanthology.org', 'aclweb.org', 'aclanthology.info'):
        p = re.sub(r'^/anthology(?:-new)?/', '/', path)
        p = re.sub(r'^/papers/', '/', p)
        segs = [s for s in p.split('/') if s]
        if not segs or segs[0] in ('events', 'volumes', 'people', 'venues', 'sigs', 'search', 'faq', 'info'):
            return None
        last = re.sub(r'\.(pdf|bib|xml)$', '', segs[-1], flags=re.I)
        m = ACL_ID.fullmatch(last)
        if m:
            return 'acl:' + m.group(1), 'acl'
        return None

    if host in ('papers.nips.cc', 'proceedings.neurips.cc', 'papers.neurips.cc'):
        m = NIPS_HASH.search(path)
        if m:
            return 'neurips:' + m.group(1).lower(), 'neurips'
        m = NIPS_LEGACY.match(path)
        if m:
            return 'neurips:n' + m.group(1), 'neurips'
        return None

    if host == 'proceedings.mlr.press':
        m = PMLR.match(path)
        if m and not m.group(2).lower().startswith(('index', 'assets')):
            return f'pmlr:{m.group(1).lower()}/{m.group(2)}', 'pmlr'
        return None

    return None


def extract(text: str):
    """Yield (paper_id, source, raw_url) for every paper reference in HN text/url."""
    if not text:
        return
    t = html.unescape(text)
    seen = set()
    cands = [clean_url(u) for u in URL_RE.findall(t)]
    cands += [clean_url(u) for u in BARE_RE.findall(t)]
    for u in cands:
        r = normalize(u)
        if r and r[0] not in seen:
            seen.add(r[0])
            yield r[0], r[1], u
    for m in ARXIV_TAG_RE.finditer(t):
        pid = 'arxiv:' + m.group(1)
        if pid not in seen:
            seen.add(pid)
            yield pid, 'arxiv', m.group(0)
