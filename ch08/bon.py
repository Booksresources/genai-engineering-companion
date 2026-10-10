import json, time, torch
from common8 import *
t0 = time.time(); m = sft_model()
N = 8; outs = sample(m, TEST, n=N, temperature=0.7, seed=3)
ok = [[brand_check(r, o)["pass"] for o in os_] for r, os_ in zip(TEST, outs)]
with torch.no_grad():
    lp = [[(seq_logprob(m, [r], [o]).item() / (len(o) + 1)) for o in os_] for r, os_ in zip(TEST, outs)]   # mean log-prob per character
res = {}
for n in [1, 2, 4, 8]:
    row = {}
    row["oracle (rule checker)"] = sum(any(o[:n]) for o in ok) / len(ok)
    row["model confidence"] = sum(o[max(range(n), key=lambda j: l[j])] for o, l in zip(ok, lp)) / len(ok)
    row["length-biased judge"] = sum(o[max(range(n), key=lambda j: len(os_[j]))] for o, os_ in zip(ok, outs)) / len(ok)
    row["mean length chosen by length judge"] = sum(max(len(x) for x in os_[:n]) for os_ in outs) / len(outs)
    res[n] = row; print(n, {k: round(v, 3) for k, v in row.items()})
res["single_sample_pass"] = sum(sum(o) for o in ok) / (len(ok) * N); res["seconds"] = time.time() - t0
print("pass rate of a single sample at T=0.7:", round(res["single_sample_pass"], 3), f"{res['seconds']:.0f}s")
json.dump(res, open("bon.json", "w"), indent=1)
