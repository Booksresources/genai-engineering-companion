"""Evaluation experiments: reference metrics vs rule checks, judge position bias, bootstrap intervals, paired comparison, pass@k vs pass^k."""
import sys, json, random, numpy as np, torch
sys.path.insert(0, "/home/claude/gb/ch8"); sys.path.insert(0, "/home/claude/gb/ch7")
from common8 import sft_model, sample, TEST, brand_check
from finetune import CLOSERS
res = {}
def lcs(a, b):
    a, b = a.split(), b.split(); d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)): d[i + 1][j + 1] = d[i][j] + 1 if a[i] == b[j] else max(d[i][j + 1], d[i + 1][j])
    return d[-1][-1]
def rouge_l(hyp, ref):
    l = lcs(hyp, ref); p, r = l / max(len(hyp.split()), 1), l / max(len(ref.split()), 1); return 0 if l == 0 else 2 * p * r / (p + r)
m = sft_model()
outs = sample(m, TEST, n=4, temperature=0.8, seed=5)                     # 240 varied outputs
rows = []
for r, os_ in zip(TEST, outs):
    ref = f"Meet the {r['product']}. {r['feature'][0].upper() + r['feature'][1:]}. {CLOSERS[0]} Northwind."
    for o in os_: rows.append({"rouge": rouge_l(o, ref), "pass": brand_check(r, o)["pass"], "name_ok": brand_check(r, o)["name"], "out": o})
rg = np.array([x["rouge"] for x in rows]); ps = np.array([x["pass"] for x in rows])
res["rouge_vs_rules"] = {"n": len(rows), "pass_rate": float(ps.mean()), "corr": float(np.corrcoef(rg, ps)[0, 1]),
                         "mean_rouge_pass": float(rg[ps].mean()), "mean_rouge_fail": float(rg[~ps].mean()),
                         "high_rouge_but_fail": float(((rg > 0.8) & ~ps).sum() / max((rg > 0.8).sum(), 1)),
                         "scatter": [(x["rouge"], int(x["pass"])) for x in rows],
                         "example_fail_high_rouge": next((x["out"] for x in rows if x["rouge"] > 0.85 and not x["pass"]), None)}
print({k: v for k, v in res["rouge_vs_rules"].items() if k != "scatter"})
# ---- bootstrap confidence intervals vs evaluation-set size, from real per-item outcomes
greedy = sample(m, TEST, n=1, temperature=0.0)
g_pass = np.array([brand_check(r, o[0])["pass"] for r, o in zip(TEST, greedy)])
bon = np.array([any(brand_check(r, o)["pass"] for o in os_) for r, os_ in zip(TEST, outs)])     # best-of-4 with the rule checker
rng = np.random.default_rng(0); res["ci"] = {}
for n in [20, 60, 200, 1000]:
    idx = rng.integers(0, 60, (2000, n)); means = ps.reshape(60, 4)[:, 0][idx].mean(1) if False else rng.binomial(n, ps.mean(), 2000) / n
    res["ci"][n] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
print("CI by size (binomial bootstrap at p=%.2f):" % ps.mean(), {k: [round(v[0], 3), round(v[1], 3)] for k, v in res["ci"].items()})
diff = bon.astype(float) - g_pass.astype(float); bs = [diff[rng.integers(0, 60, 60)].mean() for _ in range(5000)]
res["paired"] = {"greedy": float(g_pass.mean()), "best_of_4": float(bon.mean()), "diff": float(diff.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 "only_greedy_passes": int(((g_pass) & (~bon)).sum()), "only_bon_passes": int(((~g_pass) & bon).sum())}
print("paired:", res["paired"])
# ---- a simulated LLM judge with position bias: pairwise comparisons with and without order swapping
truth = rng.random(400) < 0.5                                             # True = answer A is really better
def judge(a_better, shown_first_is_a, bias=0.2, acc=0.85):
    correct = rng.random() < acc; pick_a = a_better if correct else not a_better
    first_pick = pick_a if shown_first_is_a else not pick_a
    if rng.random() < bias: first_pick = True                               # extra preference for whichever answer is shown first
    return first_pick if shown_first_is_a else not first_pick              # returns "A preferred"
one = np.array([judge(t, True) for t in truth]); swap_votes = [(judge(t, True), judge(t, False)) for t in truth]
both = np.array([a if a == b else None for a, b in swap_votes], dtype=object)
consistent = np.array([x is not None for x in both])
res["judge"] = {"single_order_accuracy": float((one == truth).mean()), "single_order_prefers_first": float(one.mean()),
                "swap_consistent_share": float(consistent.mean()), "swap_accuracy_on_consistent": float((both[consistent].astype(bool) == truth[consistent]).mean())}
print("judge:", res["judge"])
# Cohen's kappa between judge and truth labels on single order
po = (one == truth).mean(); pe = one.mean() * truth.mean() + (1 - one.mean()) * (1 - truth.mean()); res["judge"]["kappa_single"] = float((po - pe) / (1 - pe))
# ---- pass@k versus pass^k on per-task success with sampling
per_task = ps.reshape(60, 4)
res["passk"] = {k: {"pass@k": float(per_task[:, :k].any(1).mean()), "pass^k": float(per_task[:, :k].all(1).mean())} for k in [1, 2, 4]}
print("pass@k / pass^k:", res["passk"])
json.dump(res, open("evals.json", "w"), indent=1)
