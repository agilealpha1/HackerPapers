#!/usr/bin/env python3
"""Step 2 (LLM pass): label each HN comment about a paper with a cheap Claude model, batched.

  export ANTHROPIC_API_KEY=...        # never put the key in a file
  python3 pipeline/label_llm.py --dry-run          # show cost estimate and one prompt
  python3 pipeline/label_llm.py                    # submit, wait, collect (resumable)

Uses the Message Batches API (half price), 25 comments per request. Standard library only.
Output: data/labels/llm_labels.jsonl  {hn_id, paper_id, stance, spam, quote, why}
build_site_data.py uses these labels automatically when the file exists.
"""
import argparse, json, os, sys, time, urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
API = 'https://api.anthropic.com/v1/messages/batches'
STANCES = ['strong_rec', 'rec', 'reference', 'critical', 'joke', 'author', 'spam']
SYSTEM = """You label Hacker News comments that link to a research paper.
For each comment return: stance, quote, why.
stance:
- strong_rec: enthusiastic recommendation ("must read", "best paper on X", "changed how I think")
- rec: points people to the paper as worth reading or useful
- reference: cites it neutrally as a source, or just mentions it
- critical: criticizes, doubts or debunks it
- joke: meme or punchline
- author: the commenter wrote the paper and is promoting it
- spam: unrelated promotion or junk
quote: a short verbatim excerpt (under 25 words) copied exactly from the comment that shows the stance; "" if none.
why: for strong_rec/rec, at most 15 words on what a reader gets from the paper per this comment; otherwise "".
Answer with only a JSON array, one object per comment, in order: [{"i":0,"stance":"...","quote":"...","why":"..."}]"""


def call(method, url, key, body=None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.environ.get('HP_LABEL_MODEL', 'claude-haiku-4-5'))
    ap.add_argument('--per-request', type=int, default=25)
    ap.add_argument('--min-users', type=int, default=2)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--threads', action='store_true', help='also label comments in the HN threads under paper submissions '
                    '(data/step1/thread_comments.jsonl, from fetch_algolia.py --threads); used for authors and pushback')
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, 'data', 'labels'); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'llm_labels.jsonl'); state_p = os.path.join(out_dir, 'batch_state.json')

    meta = {}
    mp = os.path.join(ROOT, 'data', 'meta', 'arxiv.jsonl')
    if os.path.exists(mp):
        for l in open(mp, encoding='utf-8'):
            m = json.loads(l); meta[m['paper_id']] = m['title']
    ments = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'step1', 'mentions.jsonl'), encoding='utf-8')]
    tp = os.path.join(ROOT, 'data', 'step1', 'thread_comments.jsonl')
    if a.threads and os.path.exists(tp):
        ments += [dict(json.loads(l), kind='thread') for l in open(tp, encoding='utf-8')]
    users = {}
    for m in ments:
        users.setdefault(m['paper_id'], set()).add(m['author'])
    done = set()
    if os.path.exists(out):
        done = {(r['hn_id'], r['paper_id']) for r in map(json.loads, open(out, encoding='utf-8'))}
    todo = [m for m in ments if m['kind'] != 'submission' and m.get('text') and len(users[m['paper_id']]) >= a.min_users
            and (m['hn_id'], m['paper_id']) not in done]
    chunks = [todo[i:i + a.per_request] for i in range(0, len(todo), a.per_request)]

    def prompt(chunk):
        lines = []
        for i, m in enumerate(chunk):
            title = meta.get(m['paper_id']) or m['paper_id']
            lines.append(f'### Comment {i}\nPaper: {title}\nHN thread: {m.get("story_title") or ""}\nComment: {m["text"][:1500]}')
        return '\n\n'.join(lines)

    est_in = sum(len(prompt(c)) for c in chunks) / 3.8 + len(chunks) * 350
    print(f'{len(todo)} comments in {len(chunks)} requests, ~{est_in/1e6:.1f}M input tokens, model {a.model}')
    if a.dry_run:
        print('--- system ---\n' + SYSTEM + '\n--- first request ---\n' + (prompt(chunks[0])[:3000] if chunks else '(nothing to do)'))
        return
    key = os.environ.get('ANTHROPIC_API_KEY') or sys.exit('set ANTHROPIC_API_KEY first')

    state = json.load(open(state_p)) if os.path.exists(state_p) else {}
    if not state.get('batch_id') and chunks:
        reqs, keymap = [], {}
        for n, c in enumerate(chunks[:10000]):
            cid = f'c{n}'
            keymap[cid] = [[m['hn_id'], m['paper_id']] for m in c]
            reqs.append({'custom_id': cid, 'params': {'model': a.model, 'max_tokens': 4000, 'system': SYSTEM,
                                                      'messages': [{'role': 'user', 'content': prompt(c)}]}})
        r = json.loads(call('POST', API, key, {'requests': reqs}))
        state = {'batch_id': r['id'], 'keymap': keymap, 'texts': {f'{h}|{p}': m['text'] for c in chunks[:10000] for m in c for h, p in [(m['hn_id'], m['paper_id'])]}}
        json.dump(state, open(state_p, 'w'))
        print('submitted batch', r['id'])
    if not state.get('batch_id'):
        print('nothing to label'); return
    while True:
        b = json.loads(call('GET', f"{API}/{state['batch_id']}", key))
        print(time.strftime('%H:%M:%S'), b['processing_status'], b.get('request_counts'), flush=True)
        if b['processing_status'] == 'ended':
            break
        time.sleep(60)
    raw = call('GET', b['results_url'], key).decode()
    good = bad = 0
    with open(out, 'a', encoding='utf-8') as f:
        for line in raw.splitlines():
            r = json.loads(line); ids = state['keymap'][r['custom_id']]
            if r['result']['type'] != 'succeeded':
                bad += len(ids); continue
            txt = ''.join(c.get('text', '') for c in r['result']['message']['content'])
            try:
                arr = json.loads(txt[txt.index('['): txt.rindex(']') + 1])
            except ValueError:
                bad += len(ids); continue
            for o in arr:
                i = o.get('i')
                if not isinstance(i, int) or not 0 <= i < len(ids) or o.get('stance') not in STANCES:
                    continue
                hn_id, pid = ids[i]
                q = (o.get('quote') or '').strip()
                if q and q not in state['texts'].get(f'{hn_id}|{pid}', ''):
                    q = ''                                  # keep only verbatim quotes
                f.write(json.dumps(dict(hn_id=hn_id, paper_id=pid, stance=o['stance'], spam=o['stance'] == 'spam',
                                        quote=q, why=(o.get('why') or '')[:140]), ensure_ascii=False) + '\n')
                good += 1
    os.remove(state_p)
    print(f'labeled {good}, failed {bad}. Re-run to submit the rest / retry failures.')


if __name__ == '__main__':
    main()
