"""GenAIOps experiments: tracing a workflow, quality drift detection, A/B testing pitfalls, response caching, cost attribution."""
import sys, json, time, random, contextlib, hashlib, numpy as np
sys.path.insert(0, "/home/claude/gb/ch16"); sys.path.insert(0, "/home/claude/gb/muse/src")
res = {}
# ---- 1) minimal tracing: spans with timing, tokens, cost, and version attributes
SPANS = []; STACK = []
@contextlib.contextmanager
def span(name, **attrs):
    s = {"name": name, "parent": STACK[-1]["name"] if STACK else None, "start": time.perf_counter(), **attrs}; STACK.append(s)
    try: yield s
    finally: s["ms"] = 1000 * (time.perf_counter() - s["start"]); STACK.pop(); SPANS.append(s)
import workflow as W, shutil
shutil.rmtree("state", ignore_errors=True)
PRICE = {"in": 1.0, "out": 4.0}
def tokens(text): return max(1, len(str(text).split()))
with span("campaign_request", user="m.lee", prompt_version="social_post/v2", model="provider-llm-2026-06", release="muse-1.4.0"):
    with span("plan"): qs = W.plan("TrailMug")
    for i, q in enumerate(qs):
        with span(f"research_{i}", tool="brand_assistant") as s: r = W.research(q); s["tokens_in"], s["tokens_out"] = tokens(q) + 300, tokens(r)
    with span("draft", model="copy-lora-v3") as s: post = W.draft("TrailMug", [], 1); s["tokens_in"], s["tokens_out"] = 420, tokens(post)
    with span("brand_check"): problems = W.critic("TrailMug", post)
for s in SPANS: s["cost_usd"] = (s.get("tokens_in", 0) * PRICE["in"] + s.get("tokens_out", 0) * PRICE["out"]) / 1e6
res["trace"] = [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items() if k != "start"} for s in SPANS]
print("trace:", [(s["name"], round(s["ms"]), s.get("cost_usd")) for s in SPANS])
# ---- 2) quality monitoring: daily brand-rule pass rate, regression on day 40, CUSUM detector
rng = np.random.default_rng(0); days = 70; n = 300
p = np.where(np.arange(days) < 40, 0.92, 0.88); daily = rng.binomial(n, p) / n
def cusum(x, target, k=0.01, h=0.05):
    s, alarms, trace = 0.0, [], []
    for i, v in enumerate(x):
        s = max(0.0, s + (target - v) - k); trace.append(s)
        if s > h: alarms.append(i); s = 0.0
    return alarms, trace
alarms, ctrace = cusum(daily, target=daily[:30].mean())
threshold_alarm = next((i for i, v in enumerate(daily) if v < 0.85), None)
res["monitoring"] = {"daily": daily.tolist(), "cusum": ctrace, "cusum_first_alarm_after_change": next((a for a in alarms if a >= 40), None), "false_alarms_before": sum(a < 40 for a in alarms),
                     "fixed_threshold_0.85_first_alarm": threshold_alarm}
print("monitoring:", {k: v for k, v in res["monitoring"].items() if k not in ("daily", "cusum")})
# ---- 3) A/B testing: sample size and the danger of peeking
def n_per_arm(p1, p2, alpha_z=1.96, power_z=0.84): pb = (p1 + p2) / 2; return int(np.ceil((alpha_z + power_z) ** 2 * 2 * pb * (1 - pb) / (p2 - p1) ** 2))
res["ab"] = {"n_per_arm_30_to_32": n_per_arm(0.30, 0.32), "n_per_arm_30_to_35": n_per_arm(0.30, 0.35)}
def false_positive_rate(peeks, trials=2000, n=8000):
    fp = 0
    for _ in range(trials):
        a = rng.random(n) < 0.3; b = rng.random(n) < 0.3                                    # no real difference
        for k in np.linspace(n / peeks, n, peeks).astype(int):
            pa, pb_ = a[:k].mean(), b[:k].mean(); pp = (pa + pb_) / 2; se = np.sqrt(2 * pp * (1 - pp) / k)
            if abs(pb_ - pa) / se > 1.96: fp += 1; break
    return fp / trials
res["ab"]["false_positives"] = {pk: false_positive_rate(pk) for pk in [1, 5, 20]}
print("ab:", res["ab"])
# ---- 4) response caching for repeated requests (Zipf-distributed questions)
qs_pop = rng.zipf(1.3, 20000); cache, hits = {}, 0
for q in qs_pop:
    k = hashlib.sha256(f"brand-v2|{q}".encode()).hexdigest()
    if k in cache: hits += 1
    else: cache[k] = True
res["cache"] = {"requests": len(qs_pop), "hit_rate": hits / len(qs_pop), "unique": len(cache)}
print("cache:", res["cache"])
# ---- 5) cost attribution by feature from a month of synthetic traces
feat = {"copywriting": (12000, 900, 250), "brand Q&A": (30000, 1500, 120), "campaign kit": (2500, 6000, 1800), "scheduling agent": (4000, 3500, 600)}
res["cost_by_feature"] = {f: n_req * (ti * PRICE["in"] + to * PRICE["out"]) / 1e6 for f, (n_req, ti, to) in feat.items()}
print("cost:", res["cost_by_feature"]); json.dump(res, open("ops.json", "w"), indent=1)
