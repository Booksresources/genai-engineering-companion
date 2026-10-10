"""Start the real Muse server as a separate process and load-test it with concurrent clients."""
import asyncio, json, subprocess, sys, time, httpx, numpy as np, os
env = dict(os.environ, PYTHONPATH="/home/claude/gb/muse/src")
proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "muse.web.app:app", "--port", "8801", "--workers", "1", "--log-level", "error"], env=env, cwd="/home/claude/gb/muse")
t0 = time.time()
while True:                                                     # wait for the readiness probe
    try:
        if httpx.get("http://127.0.0.1:8801/healthz", timeout=2).status_code == 200: break
    except Exception: time.sleep(0.2)
startup = time.time() - t0
QS = ["Which hex value is Fjord Green?", "What is the hashtag limit for social posts?", "What do waterproof claims require?", "How should warranty language be phrased?"]
async def one(client, i):
    t = time.perf_counter(); r = await client.get("http://127.0.0.1:8801/api/ask", params={"q": QS[i % 4]}); return time.perf_counter() - t, r.status_code
async def run(concurrency, total=200):
    async with httpx.AsyncClient(timeout=30) as client:
        sem = asyncio.Semaphore(concurrency); lat = []
        async def task(i):
            async with sem: lat.append(await one(client, i))
        t = time.perf_counter(); await asyncio.gather(*[task(i) for i in range(total)]); wall = time.perf_counter() - t
    l = np.array([x[0] for x in lat]); return {"rps": total / wall, "p50_ms": 1000 * np.percentile(l, 50), "p95_ms": 1000 * np.percentile(l, 95), "errors": sum(x[1] != 200 for x in lat)}
res = {"startup_s": startup, "load": {}}
for c in [1, 4, 16, 64]:
    res["load"][c] = asyncio.run(run(c)); print(c, {k: round(v, 1) for k, v in res["load"][c].items()})
proc.terminate(); proc.wait(); print("startup", round(startup, 2), "s")
json.dump(res, open("load.json", "w"), indent=1)
