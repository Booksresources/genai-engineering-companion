"""Discrete-event simulations: GPU replicas with a queue and autoscaling; routing with fallbacks; cost per request."""
import json, heapq, random, numpy as np
res = {}
# ---- 1) a day of traffic served by GPU replicas (each handles 4 concurrent requests, 3 s each)
def traffic(rng, hours=24):
    t, arr = 0.0, []
    while t < hours * 3600:
        h = t / 3600; rate = 1.0 + 5.0 * np.exp(-((h - 14) / 3.5) ** 2)         # requests per second, peak mid-afternoon
        t += rng.expovariate(rate); arr.append(t)
    return arr
def simulate(policy, seed=0, slots=4, service=3.0, cold_start=90.0, fixed=6, interval=30.0):
    rng = random.Random(seed); arr = traffic(rng)
    ready = fixed if policy == "fixed" else 2; starting = []; free = ready * slots; queue = []; waits = []; gpu_s = 0.0; last = 0.0; log = []
    events = [(a, 0, "arrive") for a in arr] + ([(t, 1, "tick") for t in np.arange(0, 24 * 3600, interval)] if policy == "autoscale" else [])
    heapq.heapify(events); recent = []
    while events:
        t, _, kind = heapq.heappop(events); gpu_s += (ready + len(starting)) * (t - last); last = t
        if kind == "arrive": queue.append(t); recent.append(t)
        elif kind == "done": free += 1
        elif kind == "ready": starting.pop(0); ready += 1; free += slots
        elif kind == "tick":
            recent = [x for x in recent if x > t - 120]; rate = len(recent) / 120
            want = max(2, int(np.ceil((rate * service + len(queue)) / (slots * 0.7))))
            for _ in range(want - ready - len(starting)): starting.append(t); heapq.heappush(events, (t + cold_start, 2, "ready"))
            while ready > want and free >= slots: ready -= 1; free -= slots          # scale down idle replicas
            log.append((t / 3600, ready + len(starting)))
        while queue and free > 0:
            a = queue.pop(0); free -= 1; waits.append(t - a + service); heapq.heappush(events, (t + service, 1, "done"))
    w = np.array(waits)
    return {"p50_s": float(np.percentile(w, 50)), "p95_s": float(np.percentile(w, 95)), "gpu_hours": gpu_s / 3600, "requests": len(w), "trace": log}
for pol, kw in [("fixed (6 GPUs)", dict(policy="fixed", fixed=6)), ("fixed (4 GPUs)", dict(policy="fixed", fixed=4)), ("autoscale", dict(policy="autoscale"))]:
    r = simulate(**kw); res.setdefault("scaling", {})[pol] = r; print(pol, {k: round(v, 1) for k, v in r.items() if k != "trace"})
# ---- 2) routing and fallback across model tiers
rng = random.Random(1)
def call(model, hard):
    lat = {"small": 0.6, "large": 2.5}[model] * rng.uniform(0.8, 1.3); fail = rng.random() < {"small": 0.01, "large": 0.05}[model]   # large provider has an outage-ish 5% error rate
    ok = (not fail) and (rng.random() < ({"small": 0.97, "large": 0.98}[model] if not hard else {"small": 0.55, "large": 0.93}[model]))
    return lat, fail, ok
def serve(strategy, n=4000):
    tot_lat, good, cost = [], 0, 0.0
    for _ in range(n):
        hard = rng.random() < 0.25
        if strategy == "large only": seq = ["large"]
        elif strategy == "small only": seq = ["small"]
        elif strategy == "router": seq = ["large" if hard else "small"]                      # router classifies difficulty (assumed 90% accurate below)
        else: seq = (["large", "small"] if hard else ["small", "large"])                      # router + fallback on error
        if strategy in ("router", "router + fallback") and rng.random() < 0.1: seq = seq[::-1] if len(seq) > 1 else [("small" if seq[0] == "large" else "large")]
        t = 0.0
        for k, mdl in enumerate(seq):
            lat, fail, ok = call(mdl, hard); t += lat; cost += {"small": 0.1, "large": 1.0}[mdl]
            if not fail: good += ok; break
        tot_lat.append(t)
    return {"quality": good / n, "cost_per_request": cost / n, "p95_s": float(np.percentile(tot_lat, 95))}
res["routing"] = {s: serve(s) for s in ["small only", "large only", "router", "router + fallback"]}
for k, v in res["routing"].items(): print(k, {a: round(b, 3) for a, b in v.items()})
# ---- 3) cost per request for self-hosting
def cost_per_1k(gpu_hour_usd, tokens_per_s_per_gpu, out_tokens, utilization):
    req_per_hour = tokens_per_s_per_gpu * 3600 * utilization / out_tokens; return 1000 * gpu_hour_usd / req_per_hour
res["cost"] = {u: cost_per_1k(2.5, 2500, 300, u) for u in [0.1, 0.3, 0.6, 0.9]}
print("self-hosted $ per 1,000 requests by utilization:", {k: round(v, 3) for k, v in res["cost"].items()})
json.dump(res, open("sims.json", "w"), indent=1)
