from normalize import normalize, extract

CASES = {
    "https://arxiv.org/abs/1706.03762": "arxiv:1706.03762",
    "https://arxiv.org/abs/1706.03762v5": "arxiv:1706.03762",
    "http://arxiv.org/pdf/1706.03762v5.pdf": "arxiv:1706.03762",
    "https://arxiv.org/pdf/2401.12345": "arxiv:2401.12345",
    "https://www.arxiv.org/html/2401.12345v2": "arxiv:2401.12345",
    "https://export.arxiv.org/abs/1412.6980": "arxiv:1412.6980",
    "https://ar5iv.labs.arxiv.org/html/1512.03385": "arxiv:1512.03385",
    "https://ar5iv.org/abs/1512.03385": "arxiv:1512.03385",
    "https://arxiv.org/abs/cs/0112017": "arxiv:cs/0112017",
    "https://arxiv.org/abs/cs.AI/0112017v2": "arxiv:cs/0112017",
    "https://arxiv.org/pdf/hep-th/9901001.pdf": "arxiv:hep-th/9901001",
    "https://doi.org/10.48550/arXiv.2303.08774": "arxiv:2303.08774",
    "https://arxiv.org/list/cs.LG/recent": None,
    "https://arxiv.org/": None,
    "https://openreview.net/forum?id=YicbFdNTTy": "openreview:YicbFdNTTy",
    "https://openreview.net/pdf?id=YicbFdNTTy": "openreview:YicbFdNTTy",
    "https://openreview.net/forum?id=YicbFdNTTy&noteId=abc123": "openreview:YicbFdNTTy",
    "https://openreview.net/group?id=ICLR.cc/2024": None,
    "https://aclanthology.org/2020.acl-main.463/": "acl:2020.acl-main.463",
    "https://aclanthology.org/2020.acl-main.463.pdf": "acl:2020.acl-main.463",
    "https://aclanthology.org/P19-1001.pdf": "acl:P19-1001",
    "https://www.aclweb.org/anthology/P19-1001.pdf": "acl:P19-1001",
    "https://www.aclweb.org/anthology/papers/N/N19/N19-1423/": "acl:N19-1423",
    "https://aclanthology.org/events/acl-2020/": None,
    "https://aclanthology.org/volumes/2020.acl-main/": None,
    "https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html": "neurips:3f5ee243547dee91fbd053c1c4a845aa",
    "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf": "neurips:3f5ee243547dee91fbd053c1c4a845aa",
    "https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf": "neurips:n7181",
    "https://papers.nips.cc/book/advances-in-neural-information-processing-systems-30-2017": None,
    "http://proceedings.mlr.press/v119/chen20j.html": "pmlr:v119/chen20j",
    "http://proceedings.mlr.press/v119/chen20j/chen20j.pdf": "pmlr:v119/chen20j",
    "http://proceedings.mlr.press/v37/ioffe15.pdf": "pmlr:v37/ioffe15",
    "http://proceedings.mlr.press/v119/": None,
    "https://github.com/foo/bar": None,
    "https://arxiv.org/abs/quantph/9907009": "arxiv:quant-ph/9907009",
    "https://arxiv.org/abs/cond-mat/0102536v1": "arxiv:cond-mat/0102536",
    "https://arxiv.org/abs/notarchive/1234567": None,
}

bad = 0
for url, want in CASES.items():
    got = normalize(url)
    got = got[0] if got else None
    if got != want:
        bad += 1
        print("FAIL", url, "->", got, "want", want)

# HN comment HTML: escaped slashes, hrefs, trailing punctuation, bare forms
hn = ('See <a href="https:&#x2F;&#x2F;arxiv.org&#x2F;abs&#x2F;2005.14165" rel="nofollow">https:&#x2F;&#x2F;arxiv.org&#x2F;abs&#x2F;2005.14165</a>. '
      'Also (arxiv.org/pdf/1810.04805), and arXiv:2106.09685v2; '
      'plus https://openreview.net/forum?id=nZeVKeeFYf9.')
got = sorted(p for p, _, _ in extract(hn))
want = sorted(["arxiv:2005.14165", "arxiv:1810.04805", "arxiv:2106.09685", "openreview:nZeVKeeFYf9"])
if got != want:
    bad += 1
    print("FAIL extract", got)
print(f"{len(CASES)+1-bad}/{len(CASES)+1} passed")
