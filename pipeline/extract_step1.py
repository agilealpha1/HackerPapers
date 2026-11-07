"""Step 1: pull AI-paper mentions out of the open-index/hacker-news parquet dump.

usage: python3 extract_step1.py --hn-dir ../hn --out ../data/step1 [--work ~/hp_work]

Writes
  mentions.parquet        one row per (HN item, paper) explicit link: story submissions and comment links
  thread_comments.parquet every live comment under a story whose URL is a paper (for stance labeling)
  papers_step1.parquet/.csv  per-paper aggregates
  report.md               counts
"""
import argparse, glob, json, os, sys, time
import duckdb, pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import extract

PREFILTER = r"arxiv\.org|ar5iv|openreview\.net|aclanthology|aclweb\.org|nips\.cc|neurips\.cc|mlr\.press|10\.48550|arxiv\s*:\s*\d"

ap = argparse.ArgumentParser()
ap.add_argument('--hn-dir', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--work', default=os.path.expanduser('~/hp_work'))
ap.add_argument('--mem', default='1500MB')
ap.add_argument('--threads', type=int, default=4)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.hn_dir, 'data', '*', '*.parquet'))) + \
        sorted(glob.glob(os.path.join(a.hn_dir, 'today', '**', '*.parquet'), recursive=True))
if not files:
    sys.exit(f'no parquet files under {a.hn_dir}/data')
os.makedirs(a.out, exist_ok=True); os.makedirs(a.work, exist_ok=True)
T0 = time.time()
def log(*m): print(f'[{time.time()-T0:7.1f}s]', *m, flush=True)
log(f'{len(files)} parquet files')

con = duckdb.connect(os.path.join(a.work, 'step1.duckdb'))
con.execute(f"SET memory_limit='{a.mem}'; SET threads={a.threads}; SET temp_directory='{a.work}/tmp'; SET preserve_insertion_order=false")
con.execute("CREATE OR REPLACE TEMP VIEW hn AS SELECT * FROM read_parquet(" + json.dumps(files) + ", union_by_name=true)")

# 0. dataset totals ---------------------------------------------------------
tot = con.execute("""SELECT count(*), count(*) FILTER (type=1), count(*) FILTER (type=2),
    min(time), max(time) FROM hn""").fetchone()
log('items/stories/comments', tot)

# 1. candidate items via cheap regex prefilter --------------------------------
con.execute(f"""CREATE OR REPLACE TABLE cand AS
  SELECT id, type, "by" AS author, time, score, parent, url, title, text, descendants,
         coalesce(dead,0)::INT AS dead
  FROM hn
  WHERE type IN (1,2) AND coalesce(deleted,0)=0 AND "by" IS NOT NULL
    AND regexp_matches(lower(coalesce(url,'') || ' ' || coalesce(text,'')), '{PREFILTER}')""")
log('candidates', con.execute("SELECT count(*), count(*) FILTER (dead=1) FROM cand").fetchone())

# 2. exact URL parsing/normalization in Python ------------------------------
rows = []
rdr = con.execute("SELECT id,type,author,time,score,parent,url,title,text,dead FROM cand").to_arrow_reader(50000)
for batch in rdr:
    for r in batch.to_pylist():
        found = {}
        if r['type'] == 1 and r['url']:
            for pid, src, raw in extract(r['url']):
                found.setdefault(pid, (src, raw, 'story_url'))
        for pid, src, raw in extract(r['text']):
            found.setdefault(pid, (src, raw, 'story_text' if r['type'] == 1 else 'comment'))
        for pid, (src, raw, how) in found.items():
            rows.append(dict(item_id=r['id'], item_type='story' if r['type'] == 1 else 'comment',
                             link_in=how, author=r['author'], time=r['time'], score=r['score'],
                             parent=r['parent'], paper_id=pid, source=src, raw_url=raw[:500],
                             title=r['title'], dead=r['dead']))
log('raw mentions', len(rows))
con.register('m_raw', pa.Table.from_pylist(rows))
con.execute("CREATE OR REPLACE TABLE m AS SELECT * FROM m_raw")

# 3. comment tree: (id, parent) for every live comment -------------------------
con.execute("""CREATE OR REPLACE TABLE edges AS
  SELECT id, parent, "by" AS author FROM hn WHERE type=2 AND coalesce(deleted,0)=0 AND coalesce(dead,0)=0""")
con.execute("CREATE OR REPLACE TABLE stories AS SELECT id, score, title, descendants, time FROM hn WHERE type=1")
log('edges', con.execute("SELECT count(*) FROM edges").fetchone()[0])

# 3a. root story of every linking comment (walk up)
con.execute("CREATE OR REPLACE TABLE up AS SELECT DISTINCT item_id AS cid, parent AS cur FROM m WHERE item_type='comment'")
for depth in range(300):
    n = con.execute("SELECT count(*) FROM up WHERE cur IN (SELECT id FROM edges)").fetchone()[0]
    if n == 0: break
    con.execute("""CREATE OR REPLACE TABLE up AS
      SELECT u.cid, coalesce(e.parent, u.cur) AS cur FROM up u LEFT JOIN edges e ON e.id=u.cur""")
con.execute("""CREATE OR REPLACE TABLE m2 AS
  SELECT m.*, CASE WHEN m.item_type='story' THEN m.item_id ELSE up.cur END AS story_id
  FROM m LEFT JOIN up ON up.cid=m.item_id""")
con.execute("""CREATE OR REPLACE TABLE mentions AS
  SELECT m2.*, s.score AS story_score, s.title AS story_title, s.descendants AS story_comments
  FROM m2 LEFT JOIN stories s ON s.id=m2.story_id""")

# 3b. full discussion threads under stories whose URL is a paper (walk down)
con.execute("""CREATE OR REPLACE TABLE paper_story AS
  SELECT DISTINCT item_id AS story_id, paper_id FROM mentions WHERE link_in='story_url' AND dead=0""")
con.execute("CREATE OR REPLACE TABLE thread AS SELECT e.id, e.parent, 1 AS depth, e.parent AS story_id FROM edges e WHERE e.parent IN (SELECT story_id FROM paper_story)")
con.execute("CREATE OR REPLACE TABLE frontier AS SELECT id, story_id FROM thread")
d = 1
while True:
    d += 1
    con.execute(f"""CREATE OR REPLACE TABLE nxt AS SELECT e.id, e.parent, f.story_id
                    FROM edges e JOIN frontier f ON e.parent=f.id""")
    n = con.execute("SELECT count(*) FROM nxt").fetchone()[0]
    if n == 0 or d > 500: break
    con.execute(f"INSERT INTO thread SELECT id, parent, {d}, story_id FROM nxt")
    con.execute("CREATE OR REPLACE TABLE frontier AS SELECT id, story_id FROM nxt")
log('thread comments', con.execute("SELECT count(*) FROM thread").fetchone()[0], 'max depth', d - 1)
con.execute("""CREATE OR REPLACE TABLE thread_comments AS
  SELECT ps.paper_id, t.story_id, t.id AS comment_id, t.parent, t.depth, h."by" AS author, h.time, h.text
  FROM thread t JOIN paper_story ps USING (story_id)
  JOIN (SELECT id, "by", time, text FROM hn WHERE type=2) h ON h.id=t.id""")

# 4. per-paper aggregates ---------------------------------------------------------
con.execute("""CREATE OR REPLACE TABLE papers AS
WITH ml AS (SELECT * FROM mentions WHERE dead=0),
lk AS (SELECT paper_id, any_value(source) AS source,
         count(*) AS n_link_mentions,
         count(DISTINCT author) AS n_linkers,
         count(*) FILTER (link_in='story_url') AS n_submissions,
         count(*) FILTER (item_type='comment') AS n_comment_links,
         count(DISTINCT story_id) AS n_stories,
         max(story_score) FILTER (link_in='story_url') AS max_story_score,
         min(time) AS first_seen, max(time) AS last_seen,
         arg_max(story_title, coalesce(story_score,0)) FILTER (link_in='story_url') AS top_story_title,
         list(DISTINCT author) AS linker_set
       FROM ml GROUP BY paper_id),
th AS (SELECT paper_id, count(*) AS n_thread_comments, list(DISTINCT author) AS thread_set
       FROM thread_comments GROUP BY paper_id)
SELECT lk.* EXCLUDE (linker_set), coalesce(th.n_thread_comments,0) AS n_thread_comments,
       coalesce(len(th.thread_set),0) AS n_thread_users,
       len(list_distinct(list_concat(lk.linker_set, coalesce(th.thread_set, [])))) AS n_users_any
FROM lk LEFT JOIN th USING (paper_id)""")

# 5. write outputs ---------------------------------------------------------------
for t in ('mentions', 'thread_comments', 'papers'):
    con.execute(f"COPY {t} TO '{os.path.join(a.out, t + ('_step1' if t=='papers' else ''))}.parquet' (FORMAT parquet, COMPRESSION zstd)")
con.execute(f"COPY (SELECT * FROM papers ORDER BY n_linkers DESC, max_story_score DESC NULLS LAST) TO '{a.out}/papers_step1.csv' (HEADER)")

q = lambda s: con.execute(s).fetchone()
stats = {
  'hn_items': tot[0], 'hn_stories': tot[1], 'hn_comments': tot[2], 'span': [str(tot[3]), str(tot[4])],
  'candidate_items': q("SELECT count(*) FROM cand")[0],
  'mention_rows_live': q("SELECT count(*) FROM mentions WHERE dead=0")[0],
  'mention_rows_dead_flagged': q("SELECT count(*) FROM mentions WHERE dead=1")[0],
  'paper_stories': q("SELECT count(*) FROM paper_story")[0],
  'thread_comments': q("SELECT count(*) FROM thread_comments")[0],
  'unique_papers': q("SELECT count(*) FROM papers")[0],
  'by_source': dict(con.execute("SELECT source, count(*) FROM papers GROUP BY 1 ORDER BY 2 DESC").fetchall()),
  'papers_ge2_linkers': q("SELECT count(*) FROM papers WHERE n_linkers>=2")[0],
  'papers_ge2_linkers_by_source': dict(con.execute("SELECT source, count(*) FROM papers WHERE n_linkers>=2 GROUP BY 1 ORDER BY 2 DESC").fetchall()),
  'papers_ge2_users_any': q("SELECT count(*) FROM papers WHERE n_users_any>=2")[0],
}
json.dump(stats, open(f'{a.out}/stats.json', 'w'), indent=2, default=str)
top = con.execute("SELECT paper_id, n_linkers, n_users_any, max_story_score, top_story_title FROM papers ORDER BY n_linkers DESC, max_story_score DESC NULLS LAST LIMIT 25").fetchall()
with open(f'{a.out}/report.md', 'w') as f:
    f.write('# Step 1 report\n\n```json\n' + json.dumps(stats, indent=2, default=str) + '\n```\n\n## Top 25 by distinct linkers\n\n| paper | linkers | users (links+threads) | top story score | title |\n|---|---|---|---|---|\n')
    for r in top: f.write('| ' + ' | '.join('' if v is None else str(v).replace('|', '/') for v in r) + ' |\n')
log('done', json.dumps(stats, default=str))
