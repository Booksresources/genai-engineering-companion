import json, random, time
from common8 import *
t0 = time.time(); m = sft_model()
rng = random.Random(0); prompts = rng.sample(TRAIN, 120)
outs = sample(m, prompts, n=4, temperature=1.0, seed=1)
pairs, stats = [], {"prompts": len(prompts), "all_pass": 0, "all_fail": 0, "mixed": 0}
for r, o in zip(prompts, outs):
    ok = [brand_check(r, x)["pass"] for x in o]
    if all(ok): stats["all_pass"] += 1
    elif not any(ok): stats["all_fail"] += 1
    else:
        stats["mixed"] += 1
        pairs.append({"product": r["product"], "feature": r["feature"], "chosen": o[ok.index(True)], "rejected": o[ok.index(False)]})
base_greedy, _ = pass_rate(m, TEST); base_sampled = None
stats.update({"pairs": len(pairs), "sft_test_greedy": base_greedy, "sft_test_sampled": base_sampled, "seconds": time.time() - t0})
print(stats); print(pairs[0])
json.dump({"pairs": pairs, "stats": stats}, open("prefs.json", "w"), indent=1)
