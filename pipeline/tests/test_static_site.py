#!/usr/bin/env python3
"""Checks SEO metadata and crawl discovery in the generated static site."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

PIPE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 'https://example.test/HackerPapers/'


class StaticSiteSeo(unittest.TestCase):
    def test_metadata_robots_and_sitemap(self):
        with tempfile.TemporaryDirectory(prefix='hp_static_') as tmp:
            page = os.path.join(tmp, 'index.html')
            data = os.path.join(tmp, 'data')
            out = os.path.join(tmp, 'out')
            os.makedirs(data)
            with open(page, 'w', encoding='utf-8') as f:
                f.write('<title>HackerPapers: AI papers HN recommends</title>\n'
                        '<style>body{}</style>\n<main>fixture</main>')
            with open(os.path.join(data, 'index.json'), 'w', encoding='utf-8') as f:
                json.dump([{
                    'id': 'arxiv:2401.00001',
                    'slug': 'paper-2401-00001',
                    'title': 'A & B <Study>',
                    'year': 2024,
                    'trending': False,
                    'trend': 0,
                    'why': 'A useful paper',
                    'q': '',
                    'tl': '',
                    'pos': 2,
                    'pick': False,
                    'hn': 3,
                }], f)
            with open(os.path.join(data, 'stats.json'), 'w', encoding='utf-8') as f:
                json.dump({}, f)
            with open(os.path.join(data, 'paths.json'), 'w', encoding='utf-8') as f:
                json.dump([{
                    'id': 'intro-path',
                    'name': 'Intro path',
                    'blurb': 'An introduction.',
                    'steps': ['arxiv:2401.00001'],
                }], f)
            env = dict(os.environ, HP_BASE_URL=BASE)
            result = subprocess.run(
                [sys.executable, os.path.join(PIPE, 'make_static.py'),
                 '--page', page, '--data', data, '--out', out],
                capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)

            with open(os.path.join(out, 'index.html'), encoding='utf-8') as f:
                homepage = f.read()
            self.assertIn('<title>HackerPapers: AI papers HN recommends</title>', homepage)
            self.assertIn(f'<link rel="canonical" href="{BASE}">', homepage)
            self.assertIn('property="og:title" content="HackerPapers: AI papers HN recommends"', homepage)
            self.assertIn('name="twitter:description"', homepage)
            json_ld = homepage.split('<script type="application/ld+json">', 1)[1].split('</script>', 1)[0]
            schema = json.loads(json_ld)
            self.assertEqual(schema['@type'], 'WebSite')
            self.assertEqual(schema['url'], BASE)

            with open(os.path.join(out, 'robots.txt'), encoding='utf-8') as f:
                robots = f.read()
            self.assertIn('User-agent: *\nAllow: /', robots)
            self.assertIn(f'Sitemap: {BASE}sitemap.xml', robots)
            sitemap = ET.parse(os.path.join(out, 'sitemap.xml')).getroot()
            ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = [node.text for node in sitemap.findall('sm:url/sm:loc', ns)]
            self.assertEqual(urls, [
                BASE,
                BASE + 'p/paper-2401-00001/',
                BASE + 'path/intro-path/',
            ])

            with open(os.path.join(out, 'p', 'paper-2401-00001', 'index.html'), encoding='utf-8') as f:
                share_page = f.read()
            self.assertIn('<title>A &amp; B &lt;Study&gt; · HackerPapers</title>', share_page)


if __name__ == '__main__':
    unittest.main()
