import json, random, numpy as np
from rag import *
answerable = [q for q in QS if q["answer"]]; res = {"chunking": {}, "retrievers": {}}
for strat in ["whole document", "by heading", "sentence", "fixed", "fixed + overlap"]:
    r = Retriever(chunk(strat)); res["chunking"][strat] = {"chunks": len(r.chunks), "recall@3": recall_at_k(r, answerable, mode="hybrid"),
                                                             "avg_chars": float(np.mean([len(t) for _, t in r.chunks]))}
    print(strat, res["chunking"][strat])
r = Retriever(chunk("by heading")); rng = random.Random(0); train = rng.sample(answerable, len(answerable) // 2); test = [q for q in answerable if q not in train]
r.fit_reranker(train)
for kind in ["brand", "brand-paraphrase", "product"]:
    qs = [q for q in test if q["kind"] == kind]; row = {}
    for mode in ["bm25", "dense", "hybrid"]: row[mode] = recall_at_k(r, qs, k=1, mode=mode)
    row["hybrid + rewrite"] = recall_at_k(r, qs, k=1, mode="hybrid", rewrite_q=True)
    row["hybrid + rewrite + rerank"] = recall_at_k(r, qs, k=1, mode="hybrid+rerank", rewrite_q=True)
    res["retrievers"][kind] = {"n": len(qs), **row}; print(kind, len(qs), {k: round(v, 2) for k, v in row.items()})
allrow = {m: recall_at_k(r, test, k=1, mode=m) for m in ["bm25", "dense", "hybrid"]}
allrow["hybrid + rewrite"] = recall_at_k(r, test, k=1, mode="hybrid", rewrite_q=True); allrow["hybrid + rewrite + rerank"] = recall_at_k(r, test, k=1, mode="hybrid+rerank", rewrite_q=True)
res["retrievers"]["all test"] = {"n": len(test), **allrow}; print("all", {k: round(v, 2) for k, v in allrow.items()})
# grounded answers + abstention on test questions and unanswerable ones
unans = [q for q in QS if not q["answer"]]; rows = []
for thr in [0.0, 0.3, 0.5, 0.7]:
    ok = cited = abst_good = abst_bad = 0
    for q in test:
        a = answer(r, q["q"], threshold=thr, rewrite_q=True)
        if a["answer"] is None: abst_bad += 1; continue
        ok += q["answer"].lower() in a["answer"].lower(); cited += a["citation"] == q["doc"]
    for q in unans: abst_good += answer(r, q["q"], threshold=thr, rewrite_q=True)["answer"] is None
    rows.append({"threshold": thr, "correct": ok / len(test), "citation_correct": cited / len(test), "wrongly_abstained": abst_bad / len(test), "abstained_on_unanswerable": abst_good / len(unans)})
    print(rows[-1])
res["answers"] = rows; res["examples"] = [dict(q=q["q"], **{k: v for k, v in answer(r, q["q"], threshold=0.5, rewrite_q=True).items() if k != "chunk"}) for q in test[:3] + unans[:2]]
for e in res["examples"]: print(e)
json.dump(res, open("rag_results.json", "w"), indent=1)
