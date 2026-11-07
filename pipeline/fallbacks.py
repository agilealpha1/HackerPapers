"""Stand-ins for the build's optional dependencies, so build_site_data.py and the tests run on plain python3 + numpy.

- Sentiment: a small word-list scorer with VADER's interface (polarity_scores(text)['compound'] in -1..1).
  VADER only nudges rule-based stances for comments Claude hasn't labeled, so the effect on the site is small.
- TF-IDF: a numpy version of what build_site_data.py asks scikit-learn for (words + bigrams, English stop words,
  min_df / max_df, sublinear tf, L2 rows). Returns a dense matrix.
Use the real libraries when installed (pip install -r requirements.txt); HP_FALLBACKS=1 forces these.
"""
import math, re
import numpy as np

POS = set('good great excellent amazing awesome brilliant clear clever cool elegant enjoy enjoyed favorite fantastic fascinating '
          'helpful impressive insightful interesting love loved nice neat recommend recommended solid superb useful valuable '
          'wonderful best beautiful classic important influential landmark seminal readable accessible thanks thank '
          'like liked likes glad happy agree agreed win wins well better impressed exciting excited fun powerful novel '
          'smart free easy easier simple simpler worth perfect strong strongly incredible remarkable outstanding favourite'.split())
NEG = set('bad poor wrong flawed misleading overhyped hype useless boring confusing terrible awful weak worse worst '
          'fail failed fails broken dubious questionable unconvincing disappointing nonsense garbage meh problem problems '
          'hate hated wrongly bias biased risk risks hard difficult unfortunately sadly sad worried concern concerns doubt'.split())
NEGATE = {'not', 'no', 'never', "isn't", "wasn't", "don't", "doesn't", "didn't", 'hardly'}


class SimpleSentiment:
    def polarity_scores(self, text):
        words = re.findall(r"[a-z']+", (text or '').lower())
        s = 0.0
        for i, w in enumerate(words):
            v = 2.0 if w in POS else -2.0 if w in NEG else 0.0
            if v and any(x in NEGATE for x in words[max(0, i - 3):i]):
                v = -v * 0.7
            s += v
        return {'compound': s / math.sqrt(s * s + 15) if s else 0.0}


STOP = set('a about above after again against all am an and any are as at be because been before being below between both but by '
           'can could did do does doing down during each few for from further had has have having he her here hers him his how i if '
           'in into is it its itself just me more most my no nor not of off on once only or other our ours out over own same she '
           'should so some such than that the their them then there these they this those through to too under until up very was '
           'we were what when where which while who whom why will with would you your yours also via using use based new'.split())


class SimpleTfidf:
    def __init__(self, min_df=2, max_df=0.5, **_):
        self.min_df, self.max_df = min_df, max_df

    def fit_transform(self, docs):
        toks = []
        for d in docs:
            w = [x for x in re.findall(r'(?u)\b\w\w+\b', d.lower()) if x not in STOP]
            toks.append(w + [a + ' ' + b for a, b in zip(w, w[1:])])
        df = {}
        for t in toks:
            for x in set(t):
                df[x] = df.get(x, 0) + 1
        n = len(docs)
        vocab = {x: i for i, x in enumerate(sorted(x for x, c in df.items() if c >= self.min_df and c <= self.max_df * n))}
        X = np.zeros((n, max(1, len(vocab))), np.float32)
        for r, t in enumerate(toks):
            for x in t:
                j = vocab.get(x)
                if j is not None:
                    X[r, j] += 1
        X = np.where(X > 0, 1 + np.log(np.maximum(X, 1)), 0).astype(np.float32)
        idf = np.array([math.log((1 + n) / (1 + df[x])) + 1 for x in sorted(vocab, key=vocab.get)] or [1], np.float32)
        X *= idf
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        return X
