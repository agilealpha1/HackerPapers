#!/usr/bin/env python3
"""End-to-end checks for the site build on a small, hand-made dataset.

  python3 -m unittest discover -s pipeline/tests -v        (or: python3 pipeline/tests/test_build.py)

Builds a fixture (mentions, arXiv metadata, Semantic Scholar rows, Claude labels, thread comments and a curated
path) in a temp folder, runs build_site_data.py on it once, and checks the rules that decide what the site shows:
the recommendation bar, the AI filter, stance fallback rules, venue-to-arXiv merging, the lasting / spike /
debated badges, thread comments, curated paths, and the output schema. make_path.py's argument checks too.
"""
import json, os, shutil, subprocess, sys, tempfile, unittest
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)


def ts(y, mo, d=15):
    return int(datetime(y, mo, d, 12, tzinfo=timezone.utc).timestamp())


def mention(hn, pid, author, t, kind='comment', text=None, title='Some HN thread', points=None, story=None):
    return dict(hn_id=hn, kind=kind, paper_id=pid, source=pid.split(':')[0], author=author, time=t,
                month=datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m'), story_id=story or hn, story_title=title,
                points=points, num_comments=None, own_chars=len(text or ''), substantive=len(text or '') >= 100,
                raw_url=f'https://arxiv.org/abs/{pid.split(":", 1)[1]}' if pid.startswith('arxiv:') else 'https://example.org/x',
                text=text)


def meta(aid, title, cats, year=2020, abstract=None):
    return dict(paper_id='arxiv:' + aid, title=title, authors=['Ada Lovelace', 'Alan Turing'],
                abstract=abstract or f'We study {title.lower()}. Neural language models and networks are trained and evaluated.',
                published=f'{year}-03-01T00:00:00Z', updated=f'{year}-03-01T00:00:00Z', primary_category=cats[0],
                categories=cats, comment=None, journal_ref=None, doi=None)


LONG = ' This comment is long enough to count as substantive commentary on the paper and its results in detail.'


class BuildFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix='hp_test_')
        R = cls.root
        for d in ('data/step1', 'data/meta', 'data/labels', 'site/data'):
            os.makedirs(os.path.join(R, d), exist_ok=True)
        M, L, n = [], [], [1000]

        def add(pid, author, t, stance=None, **kw):
            n[0] += 1
            M.append(mention(n[0], pid, author, t, **kw))
            if stance:
                L.append(dict(hn_id=n[0], paper_id=pid, stance=stance, quote='', why='good intro' if stance in ('rec', 'strong_rec') else ''))
            return n[0]
        # 1. lasting: recommended in 5 months over 5 years
        for i, (y, mo) in enumerate([(2019, 3), (2020, 6), (2021, 9), (2023, 1), (2024, 5)]):
            add('arxiv:2001.00001', f'u{i}', ts(y, mo), 'rec', text='Good one.' + LONG)
        # 2. spike: 6 of 8 mentions in one month, recommended only in 2026
        for i in range(6):
            add('arxiv:2002.00002', f's{i}', ts(2026, 1, 3 + i), 'rec' if i < 3 else 'reference', text='Neat.' + LONG)
        add('arxiv:2002.00002', 's6', ts(2026, 2), 'reference', text='Seen it.')
        add('arxiv:2002.00002', 's7', ts(2026, 2, 20), 'reference', text='Seen it too.')
        # 3. debated: 2 recommendations and 2 pieces of pushback
        add('arxiv:2003.00003', 'd1', ts(2025, 1), 'rec', text='Read it.' + LONG)
        add('arxiv:2003.00003', 'd2', ts(2025, 2), 'rec', text='Read it.' + LONG)
        add('arxiv:2003.00003', 'd3', ts(2025, 3), 'critical', text='Flawed.' + LONG)
        add('arxiv:2003.00003', 'd4', ts(2025, 4), 'critical', text='Not convinced.' + LONG)
        # 4. below the bar: one recommender (kept only because a curated path lists it)
        add('arxiv:2004.00004', 'b1', ts(2024, 1), 'rec', text='Worth it.' + LONG)
        # 5. not AI (math.CO), two recommenders: dropped
        add('arxiv:2005.00005', 'm1', ts(2024, 2), 'rec', text='Fun.' + LONG)
        add('arxiv:2005.00005', 'm2', ts(2024, 3), 'rec', text='Fun.' + LONG)
        # 6. NeurIPS copy of paper 1, matched by story title
        add('neurips:abc123', 'v1', ts(2022, 2), kind='submission', title='Lasting Transformers for Language Modeling', points=50)
        add('neurips:abc123', 'v2', ts(2022, 2, 20), 'rec', text='Great.' + LONG)
        # 7. OpenReview link whose S2 record names an arXiv id HN never linked: canonicalized to it
        add('openreview:XYZ', 'o1', ts(2023, 5), 'rec', text='Recommended.' + LONG)
        add('openreview:XYZ', 'o2', ts(2023, 6), 'rec', text='Recommended.' + LONG)
        # 8. no Claude labels: the v0 rules decide (strong recommendations, and an author)
        add('arxiv:2007.00007', 'r1', ts(2024, 6), text='This is a must read for anyone working on neural networks.' + LONG)
        add('arxiv:2007.00007', 'r2', ts(2024, 7), text='I highly recommend this, a classic paper on neural networks.' + LONG)
        add('arxiv:2007.00007', 'r3', ts(2024, 8), text='Author here, our paper shows the rule works; happy to answer questions.' + LONG)
        sub3 = add('arxiv:2003.00003', 'd0', ts(2025, 1, 2), kind='submission', title='Debated Language Model Benchmarks', points=120)
        cls.n_items = n[0]
        with open(os.path.join(R, 'data/step1/mentions.jsonl'), 'w') as f:
            f.writelines(json.dumps(m) + '\n' for m in M)
        with open(os.path.join(R, 'data/labels/llm_labels.jsonl'), 'w') as f:
            f.writelines(json.dumps(r) + '\n' for r in L)
        # thread comments under paper 3's submission: an author (rules) and Claude-labeled pushback
        T = [dict(hn_id=9001, paper_id='arxiv:2003.00003', story_id=sub3, depth=1, author='authorx', time=ts(2025, 1, 3), month='2025-01',
                  own_chars=150, substantive=True, text="I'm one of the authors, happy to answer questions about our benchmark." + LONG),
             dict(hn_id=9002, paper_id='arxiv:2003.00003', story_id=sub3, depth=1, author='skeptic', time=ts(2025, 1, 4), month='2025-01',
                  own_chars=150, substantive=True, text='The evaluation is cherry-picked.' + LONG)]
        with open(os.path.join(R, 'data/step1/thread_comments.jsonl'), 'w') as f:
            f.writelines(json.dumps(t) + '\n' for t in T)
        with open(os.path.join(R, 'data/labels/llm_labels.jsonl'), 'a') as f:
            f.write(json.dumps(dict(hn_id=9002, paper_id='arxiv:2003.00003', stance='critical', quote='cherry-picked', why='')) + '\n')
        MD = [meta('2001.00001', 'Lasting Transformers for Language Modeling', ['cs.CL'], 2019),
              meta('2002.00002', 'Spiky Neural Diffusion Models', ['cs.LG'], 2025),
              meta('2003.00003', 'Debated Language Model Benchmarks', ['cs.CL'], 2025),
              meta('2004.00004', 'Rarely Mentioned Neural Network Training', ['cs.LG'], 2023),
              meta('2005.00005', 'Magic Card Game Rules', ['math.CO'], 2024, abstract='We count cards in a game.'),
              meta('2006.00006', 'Canonical Neural Language Networks', ['cs.LG'], 2023),
              meta('2007.00007', 'Neural Rule Fallback Networks', ['cs.LG'], 2024),
              meta('2008.00008', 'Editor Pick Neural Language Models', ['cs.LG'], 2023)]
        with open(os.path.join(R, 'data/meta/arxiv.jsonl'), 'w') as f:
            f.writelines(json.dumps(m) + '\n' for m in MD)
        with open(os.path.join(R, 'data/meta/s2.jsonl'), 'w') as f:
            f.write(json.dumps(dict(paper_id='openreview:XYZ', s2_id='s2xyz', title='Canonical Neural Language Networks', year=2023,
                                    venue='ICLR', citations=42, influential=3, arxiv_id='2006.00006', doi=None, url='https://s2/x')) + '\n')
        json.dump([dict(id='test-path', name='Test path', blurb='A test.', steps=[
            dict(id='arxiv:2001.00001', hn=6, core=True, stage=0), dict(id='arxiv:2004.00004', hn=1, why='below the bar', stage=0),
            dict(id='arxiv:2008.00008', hn=0, pick=True, stage=1)], stages=[dict(name='One'), dict(name='Two')])],
            open(os.path.join(R, 'data/curated_paths.json'), 'w'))
        out = os.path.join(R, 'site/data')
        r = subprocess.run([sys.executable, os.path.join(PIPE, 'build_site_data.py'), '--root', R, '--out', out],
                           capture_output=True, text=True)
        cls.log = r.stdout + r.stderr
        if r.returncode:
            raise RuntimeError('build failed:\n' + cls.log[-3000:])
        cls.index = json.load(open(os.path.join(out, 'index.json')))
        cls.by = {x['id']: x for x in cls.index}
        cls.shard = {}
        for fn in os.listdir(os.path.join(out, 'p')):
            cls.shard.update(json.load(open(os.path.join(out, 'p', fn))))
        cls.paths = json.load(open(os.path.join(out, 'paths.json')))
        cls.stats = json.load(open(os.path.join(out, 'stats.json')))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_bar_and_ai_filter(self):
        self.assertIn('arxiv:2001.00001', self.by)
        self.assertNotIn('arxiv:2005.00005', self.by, 'non-AI paper must be dropped')
        self.assertTrue(self.by['arxiv:2004.00004'].get('xp'), 'below-bar paper is path-only')
        self.assertEqual(self.stats['papers'], sum(1 for x in self.index if not x.get('xp')))
        self.assertEqual(self.stats['path_only'], 2)

    def test_venue_merges(self):
        self.assertNotIn('neurips:abc123', self.by, 'NeurIPS copy merges into the arXiv record by title')
        self.assertEqual(self.by['arxiv:2001.00001']['pos'], 7, '5 + the NeurIPS submitter (50 points) + its recommender')
        self.assertNotIn('openreview:XYZ', self.by)
        self.assertIn('arxiv:2006.00006', self.by, 'OpenReview link canonicalized to the arXiv id from S2')
        self.assertEqual(self.by['arxiv:2006.00006']['cites'], 42, 'citations follow the canonical id')

    def test_badges(self):
        self.assertEqual(self.by['arxiv:2001.00001']['lasting'], 'lasting')
        self.assertEqual(self.by['arxiv:2002.00002']['lasting'], 'spike')
        self.assertTrue(self.by['arxiv:2003.00003']['deb'])
        self.assertFalse(self.by['arxiv:2001.00001']['deb'])

    def test_rule_fallback_stances(self):
        p = self.by['arxiv:2007.00007']
        self.assertEqual(p['strong'], 2, 'must-read / highly recommend -> strong_rec without Claude labels')
        self.assertTrue(p['auth'], '"Author here, our paper" -> author on HN')

    def test_thread_comments(self):
        d = self.shard['arxiv:2003.00003']
        self.assertTrue(any(a['id'] == 9001 for a in d['authors_hn']), 'author found in the submission thread')
        self.assertTrue(any(x['id'] == 9002 for x in d['pushback']), 'Claude-labeled thread pushback shown')
        self.assertEqual(self.by['arxiv:2003.00003']['pos'], 3, 'd1, d2 and the 120-point submitter; thread comments never count')

    def test_curated_path(self):
        cp = next(p for p in self.paths if p.get('curated'))
        self.assertEqual(cp['steps'], ['arxiv:2001.00001', 'arxiv:2004.00004', 'arxiv:2008.00008'])
        self.assertTrue(cp['info']['arxiv:2008.00008'].get('pick'))
        self.assertEqual(cp['info']['arxiv:2004.00004']['stage'], 0)
        self.assertTrue(self.by['arxiv:2008.00008'].get('pick') and self.by['arxiv:2008.00008'].get('xp'))
        self.assertEqual(len(cp['stages']), 2)
        self.assertTrue(any(x['id'] == 'test-path' for x in self.shard['arxiv:2004.00004']['paths']))

    def test_schema(self):
        need = {'id', 'title', 'year', 'topics', 'level', 'users', 'pos', 'score', 'slug', 'sh', 'au', 'code', 'lasting', 'deb', 'auth'}
        for r in self.index:
            self.assertFalse(need - set(r), f'{r["id"]} lacks {need - set(r)}')
            self.assertIn(r['id'], self.shard)
        self.assertEqual(len({r['slug'] for r in self.index}), len(self.index), 'slugs are unique')
        for p in self.paths:
            for s in p['steps']:
                self.assertIn(s, self.by, f'path {p["id"]} step {s} is in the index')


class MakePathArgs(unittest.TestCase):
    def run_mp(self, *args):
        return subprocess.run([sys.executable, os.path.join(PIPE, 'make_path.py'), *args], capture_output=True, text=True)

    def test_placeholders_rejected(self):
        r = self.run_mp('--topic', 'Name: description', '--seeds', 'id1,id2', '--keywords', 'a;b')
        self.assertEqual(r.returncode, 1)
        for s in ('not arXiv ids: id1, id2', "at least 3 characters: 'a', 'b'", 'placeholder'):
            self.assertIn(s, r.stderr)

    def test_old_style_ids_accepted_by_check(self):
        r = self.run_mp('--topic', 'x', '--seeds', 'cs/0112017v2', '--keywords', 'hypernetwork')
        self.assertNotIn('not arXiv ids', r.stderr)


if __name__ == '__main__':
    unittest.main()
