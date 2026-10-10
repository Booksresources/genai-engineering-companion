import json, os, shutil, time, random
from workflow import *
shutil.rmtree("state", ignore_errors=True); res = {}
# 1) sequential versus parallel research workers
for par in [False, True]:
    shutil.rmtree("state", ignore_errors=True); r = campaign("seq" if not par else "par", parallel=par)
    res["parallel" if par else "sequential"] = r["seconds"]; print("parallel" if par else "sequential", round(r["seconds"], 2), "s; critic rounds", r["critic_rounds"])
res["final_post"] = r["post"]; res["critic_rounds"] = r["critic_rounds"]
# 2) durable execution: crash after research, then resume
shutil.rmtree("state", ignore_errors=True)
try: campaign("durable", crash_after="research")
except SystemExit: pass
t0 = time.time(); r = campaign("durable"); res["resume"] = {"seconds": time.time() - t0, "log": r["log"]}
print("resume", round(res["resume"]["seconds"], 2), "s;", [l for l in r["log"] if l.startswith("skip")][:6])
# 3) human-in-the-loop: pause for approval, resume when the decision arrives
shutil.rmtree("state", ignore_errors=True)
try: campaign("hitl", decision=None)
except NeedsApproval as e: res["paused"] = str(e); print("paused:", e)
r = campaign("hitl", decision="approved"); res["after_approval"] = r["approval"]; print("after approval:", r["approval"], [l for l in r["log"] if l.startswith("ran")][-2:])
# 4) worker failures with and without retries (parallel workflow, 30% timeout per worker call)
for retries in [0, 2]:
    ok = 0
    for trial in range(60):
        shutil.rmtree("state", ignore_errors=True); rng = random.Random(trial); wf = Workflow(f"r{trial}")
        try:
            [wf.step(f"research_{i}", research, q, 0.3, rng, retries=retries) for i, q in enumerate(QUESTIONS["TrailMug"])]; ok += 1
        except Exception: pass
    res[f"retries_{retries}"] = ok / 60; print("retries", retries, "success", ok / 60)
json.dump(res, open("wf.json", "w"), indent=1)
