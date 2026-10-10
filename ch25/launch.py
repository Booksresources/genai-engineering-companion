"""Capstone: run the integration suite, then simulate a staged rollout in which a new prompt version causes an incident."""
import json, random, subprocess, sys, numpy as np
sys.path.insert(0, "/home/claude/gb/muse/src")
from muse.safety.layer import check_output
res = {}
p = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd="/home/claude/gb/muse", capture_output=True, text=True)
res["tests"] = p.stdout.strip().splitlines()[-1]; print("integration:", res["tests"])
rng = random.Random(0)
GOOD = ["The {p} keeps drinks hot for 6 hours.", "Meet the {p}, built for long days outside.", "The {p} packs down small for any trip."]
BAD = ["The {p} cures cold hands for good.", "Get the {p} at half price today.", "The {p} is 100 percent waterproof."]
def post(version):
    bad_rate = {"v2": 0.01, "v3": 0.08}[version]                     # v3 is a new prompt that occasionally drifts into claims and prices
    t = rng.choice(BAD if rng.random() < bad_rate else GOOD).format(p="TrailMug")
    return {"headline": "Hot coffee, cold trails", "body": t, "hashtags": ["#Northwind"]}
def violated(version): return bool(check_output(post(version), "TrailMug", ["keeps drinks hot for 6 hours"]))
stages = [("canary", 0.05, 4), ("10%", 0.10, 8), ("50%", 0.50, 12), ("100%", 1.0, 24)]          # share of traffic, hours
def rollout(new, per_hour=400, guard=0.03, min_n=100):
    exposed_bad, timeline = 0, []
    for name, share, hours in stages:
        n_new = bad_new = n_old = bad_old = 0
        for h in range(hours):
            for _ in range(per_hour):
                if rng.random() < share: n_new += 1; v = violated(new); bad_new += v; exposed_bad += v
                else: n_old += 1; bad_old += violated("v2")
            rate_new = bad_new / max(n_new, 1); rate_old = bad_old / max(n_old, 1)
            timeline.append({"stage": name, "hour": len(timeline), "new_rate": rate_new, "old_rate": rate_old, "n_new": n_new})
            if n_new >= min_n and rate_new - rate_old > guard:                # automatic rollback guard
                return {"rolled_back": True, "stage": name, "hours": len(timeline), "violations_shown_to_reviewers": exposed_bad, "timeline": timeline}
    return {"rolled_back": False, "stage": "100%", "hours": len(timeline), "violations_shown_to_reviewers": exposed_bad, "timeline": timeline}
res["rollout_v3"] = rollout("v3"); res["rollout_v2_rerelease"] = rollout("v2")
big_bang = sum(violated("v3") for _ in range(400 * 48)); res["big_bang_v3_violations_48h"] = big_bang
print("v3 staged:", {k: v for k, v in res["rollout_v3"].items() if k != "timeline"}); print("v2 staged:", {k: v for k, v in res["rollout_v2_rerelease"].items() if k != "timeline"}); print("big bang v3, 48h:", big_bang)
json.dump(res, open("launch.json", "w"), indent=1)
