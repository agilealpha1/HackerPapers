#!/usr/bin/env python3
"""Wrap site/index.html (artifact-style page body) into a standalone static site in site_dist/.
usage: python3 pipeline/make_static.py [--data site/data] [--out site_dist]"""
import argparse, os, shutil
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ap = argparse.ArgumentParser()
ap.add_argument('--page', default=os.path.join(ROOT, 'site', 'index.html'))
ap.add_argument('--data', default=os.path.join(ROOT, 'site', 'data'))
ap.add_argument('--out', default=os.path.join(ROOT, 'site_dist'))
a = ap.parse_args()
body = open(a.page, encoding='utf-8').read()
head_end = body.index('</style>') + len('</style>')
BASE = os.environ.get('HP_BASE_URL', 'https://hackerpapers.example/').rstrip('/') + '/'
from html import escape as h
SITE_DESC = 'The AI research papers Hacker News keeps recommending, with what HN actually said about them.'
doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
       '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
       f'<meta name="description" content="{h(SITE_DESC)}">\n<meta name="hp-share" content="1">\n'
       f'<meta property="og:type" content="website"><meta property="og:site_name" content="HackerPapers">'
       f'<meta property="og:title" content="HackerPapers: AI papers HN recommends"><meta property="og:description" content="{h(SITE_DESC)}">'
       f'<meta property="og:url" content="{h(BASE)}"><meta name="twitter:card" content="summary">\n'
       '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>\n'
       + body[:head_end] + '\n</head>\n<body>\n' + body[head_end:] + '\n</body>\n</html>\n')
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, 'index.html'), 'w', encoding='utf-8').write(doc)
if os.path.abspath(a.data) != os.path.abspath(os.path.join(a.out, 'data')):
    shutil.copytree(a.data, os.path.join(a.out, 'data'), dirs_exist_ok=True)   # overwrite in place
# Atom feed: the 30 papers trending on HN in the last 12 months
import json
from datetime import datetime, timezone
from xml.sax.saxutils import escape as x
idx = json.load(open(os.path.join(a.data, 'index.json'), encoding='utf-8'))
stats = json.load(open(os.path.join(a.data, 'stats.json'), encoding='utf-8'))
base = os.environ.get('HP_BASE_URL', 'https://hackerpapers.example/').rstrip('/') + '/'
now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def link(r):
    src, rest = r['id'].split(':', 1)
    return f'https://arxiv.org/abs/{rest}' if src == 'arxiv' else (r.get('url') or base)
items = sorted([r for r in idx if r.get('trending')], key=lambda r: -r['trend'])[:30]
feed = ['<?xml version="1.0" encoding="utf-8"?>', '<feed xmlns="http://www.w3.org/2005/Atom">',
        '<title>HackerPapers: trending this year</title>', f'<id>{x(base)}feed.xml</id>', f'<updated>{now}</updated>',
        f'<link rel="alternate" href="{x(base)}#trending"/>', f'<link rel="self" href="{x(base)}feed.xml"/>',
        '<author><name>HackerPapers</name></author>']
for r in items:
    summary = (r.get('why') or r.get('q') or r.get('tl') or '')
    feed += ['<entry>', f"<title>{x(r['title'])}</title>", f"<id>{x(base)}#paper-{x(r['slug'])}</id>",
             f'<link rel="alternate" href="{x(base)}#paper-{x(r["slug"])}"/>', f'<link rel="related" href="{x(link(r))}"/>',
             f'<updated>{now}</updated>',
             f"<summary>{x(summary)} ({r['pos']} HN recommendations, {r['users']} HN people)</summary>", '</entry>']
feed.append('</feed>')
open(os.path.join(a.out, 'feed.xml'), 'w', encoding='utf-8').write('\n'.join(feed) + '\n')
# share pages: /p/<slug>/ and /path/<id>/ carry per-page titles and OpenGraph/Twitter tags for link previews,
# then send the browser on to the app (#paper-<slug>, #path-<id>). Hash URLs can't carry these tags.
def share_page(path, title, desc, target, kind='article'):
    d = os.path.join(a.out, path); os.makedirs(d, exist_ok=True)
    url = BASE + path + '/'
    open(os.path.join(d, 'index.html'), 'w', encoding='utf-8').write(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
        f'<title>{h(title)} · HackerPapers</title>\n<meta name="description" content="{h(desc)}">\n'
        f'<link rel="canonical" href="{h(url)}">\n<meta property="og:type" content="{kind}"><meta property="og:site_name" content="HackerPapers">\n'
        f'<meta property="og:title" content="{h(title)}"><meta property="og:description" content="{h(desc)}"><meta property="og:url" content="{h(url)}">\n'
        f'<meta name="twitter:card" content="summary"><meta name="twitter:title" content="{h(title)}"><meta name="twitter:description" content="{h(desc)}">\n'
        f'<meta http-equiv="refresh" content="0; url=../../{h(target)}">\n'
        f'</head><body><p><a href="../../{h(target)}">{h(title)}</a> on HackerPapers</p></body></html>\n')
for sub in ('p', 'path'):
    if os.path.isdir(os.path.join(a.out, sub)):
        shutil.rmtree(os.path.join(a.out, sub))
for r in idx:
    say = r.get('why') or r.get('q') or r.get('tl') or ''
    hn = f"{r['pos']} HN recommendations" if r.get('pos') else ("Editor's pick" if r.get('pick') else f"{r.get('hn', 0)} HN mentions")
    desc = (f'{say[:220]}{"…" if len(say) > 220 else ""} ({hn}, {r["year"]})').strip()
    share_page(f"p/{r['slug']}", r['title'], desc, f"#paper-{r['slug']}")
paths_p = os.path.join(a.data, 'paths.json')
for t in (json.load(open(paths_p, encoding='utf-8')) if os.path.exists(paths_p) else []):
    share_page(f"path/{t['id']}", f"Reading path: {t['name']}", f"{t['blurb']} {len(t['steps'])} papers.", f"#path-{t['id']}")
print(f'share pages: {len(idx)} papers, /p/<slug>/ and /path/<id>/', flush=True)
hdr = os.path.join(ROOT, 'deploy', '_headers')
if os.path.exists(hdr):
    shutil.copy(hdr, os.path.join(a.out, '_headers'))
print('static site ->', os.path.abspath(a.out))
