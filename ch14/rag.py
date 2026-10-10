"""RAG components: chunking, BM25, dense (LSA) retrieval, hybrid RRF, a learned reranker, query rewriting,
extractive grounded answers with citations, and abstention."""
import json, math, re, numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
KB = json.load(open("/home/claude/gb/ch14/kb.json")); DOCS, QS = KB["docs"], KB["questions"]
STOP = set("the a an of to is are in for on and or be with what how which does do can we our must should used use".split())
def toks(s): return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]

# ---------- chunking strategies ----------
def chunk(strategy, size=160, overlap=60):
    out = []
    for d in DOCS:
        sents = [s for sec in d["sections"] for s in sec["sentences"]]
        if strategy == "whole document": out.append((d["id"], d["title"] + ". " + " ".join(sents)))
        elif strategy == "by heading":
            for sec in d["sections"]: out.append((d["id"], d["title"] + " > " + sec["heading"] + ". " + " ".join(sec["sentences"])))
        elif strategy == "sentence": out += [(d["id"], s) for s in sents]
        else:
            text = " ".join(sents); step = size - (overlap if strategy == "fixed + overlap" else 0)
            out += [(d["id"], text[i:i + size]) for i in range(0, max(len(text) - (overlap if strategy == "fixed + overlap" else 0), 1), step)]
    return out

# ---------- retrievers ----------
class BM25:
    def __init__(s, texts, k1=1.5, b=0.75):
        s.docs = [Counter(toks(t)) for t in texts]; s.len = np.array([sum(d.values()) for d in s.docs]); s.avg = s.len.mean(); s.k1, s.b = k1, b
        df = Counter(w for d in s.docs for w in d); N = len(texts); s.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    def scores(s, q):
        qt = toks(q); out = np.zeros(len(s.docs))
        for i, d in enumerate(s.docs):
            for w in qt:
                if w in d: f = d[w]; out[i] += s.idf[w] * f * (s.k1 + 1) / (f + s.k1 * (1 - s.b + s.b * s.len[i] / s.avg))
        return out
class Dense:
    """Stand-in for a neural embedding model: TF-IDF over words and character n-grams, reduced with SVD (LSA)."""
    def __init__(s, texts, dim=96):
        s.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(texts)
        X = s.vec.transform(texts); s.svd = TruncatedSVD(min(dim, X.shape[1] - 1), random_state=0).fit(X)
        s.E = s._norm(s.svd.transform(X))
    def _norm(s, E): return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    def embed(s, texts): return s._norm(s.svd.transform(s.vec.transform(texts)))
    def scores(s, q): return s.E @ s.embed([q])[0]
def rrf(*score_lists, k=60):
    out = np.zeros(len(score_lists[0]))
    for sc in score_lists:
        ranks = np.argsort(-sc).argsort(); out += 1 / (k + ranks + 1)
    return out
REWRITES = {"exclamation point": "exclamation marks", "font": "typeface", "titles": "headlines", "padding": "clear space", "logo mark": "logo",
            "hashtag limit": "how many hashtags", "accent shade": "accent color", "buttons": "calls to action", "paragraph text": "body copy",
            "synthetic photos": "ai-generated images", "disclosure": "labeled", "guarantee": "warranty", "evidence backs": "require",
            "website": "on screen", "small": "minimum width", "large": "size", "allowed": "post"}
def rewrite(q):
    """Rule-based stand-in for LLM query rewriting: map user wording to the vocabulary of the documents."""
    out = q.lower()
    for a, b in REWRITES.items(): out = out.replace(a, b)
    return out

class Retriever:
    def __init__(s, chunks):
        s.chunks = chunks; texts = [t for _, t in chunks]; s.bm25 = BM25(texts); s.dense = Dense(texts); s.reranker = None
    def features(s, q, idx, b, d):
        qt = set(toks(q)); feats = []
        for i in idx:
            ct = set(toks(s.chunks[i][1])); feats.append([b[i], d[i], len(qt & ct) / max(len(qt), 1), float(any(w in s.chunks[i][1].lower() for w in re.findall(r"[A-Z][a-z]+[A-Z][a-z]+", q.replace("the ", ""))))])
        return np.array(feats)
    def search(s, q, mode="hybrid", k=3, rewrite_q=False, pool=20):
        if rewrite_q: q = rewrite(q)
        b, d = s.bm25.scores(q), s.dense.scores(q)
        sc = {"bm25": b, "dense": d, "hybrid": rrf(b, d)}["hybrid" if mode == "hybrid+rerank" else mode]
        idx = list(np.argsort(-sc)[:pool])
        if mode == "hybrid+rerank" and s.reranker is not None:
            p = s.reranker.predict_proba(s.features(q, idx, b, d))[:, 1]; idx = [idx[i] for i in np.argsort(-p)]; top = float(p.max())
        else: top = float(sc[idx[0]])
        return idx[:k], top
    def fit_reranker(s, questions, rewrite_q=True):
        X, y = [], []
        for qq in questions:
            q = rewrite(qq["q"]) if rewrite_q else qq["q"]; b, d = s.bm25.scores(q), s.dense.scores(q); idx = list(np.argsort(-rrf(b, d))[:20])
            X.append(s.features(q, idx, b, d)); y += [int(s.chunks[i][0] == qq["doc"] and (qq["answer"] or "").lower() in s.chunks[i][1].lower()) for i in idx]
        s.reranker = LogisticRegression(max_iter=2000).fit(np.vstack(X), y)

def recall_at_k(ret, questions, k=3, **kw):
    hits = [any(ret.chunks[i][0] == q["doc"] and q["answer"].lower() in ret.chunks[i][1].lower() for i in ret.search(q["q"], k=k, **kw)[0]) for q in questions]
    return float(np.mean(hits))

def answer(ret, q, k=3, threshold=0.5, **kw):
    """Extractive grounded answer: pick the retrieved sentence that best matches the question, cite its chunk; abstain if unsure."""
    idx, top = ret.search(q, k=k, mode="hybrid+rerank", **kw)
    if top < threshold: return {"answer": None, "citation": None, "confidence": top}
    best, best_s = None, -1
    for i in idx:
        for sent in re.split(r"(?<=\.)\s+", ret.chunks[i][1]):
            sc = len(set(toks(rewrite(q))) & set(toks(sent))) + (2 if any(n in sent for n in re.findall(r"[A-Z][a-z]+[A-Z][a-z]+", q)) else 0)
            if sc > best_s: best, best_s = (sent.strip(), i), sc
    return {"answer": best[0], "citation": ret.chunks[best[1]][0], "chunk": best[1], "confidence": top}
