import json, time, torch
from diff9 import *
from metrics9 import score
betas, abar = schedule("cosine", 200)
m = UNet(16, 10); m.load_state_dict(torch.load("unet_digits.pt")); m.eval()
y = torch.arange(500) % 10; res = {"samplers": {}, "guidance": {}}
for name, steps in [("DDPM 200", None), ("DDIM 50", 50), ("DDIM 20", 20), ("DDIM 10", 10), ("DDIM 5", 5)]:
    t0 = time.time(); S = sample(m, 500, y, betas, abar, steps=steps); dt = time.time() - t0
    r = score(S, y); r["seconds"] = dt; res["samplers"][name] = r; print(name, {k: round(v, 3) for k, v in r.items()})
for w in [1.0, 2.0, 3.0, 5.0]:
    S = sample(m, 500, y, betas, abar, steps=50, guidance=w); r = score(S, y); res["guidance"][w] = r
    if w in (1.0, 3.0): torch.save(S[:40], f"samples_w{w}.pt")
    print("guidance", w, {k: round(v, 3) for k, v in r.items()})
d = DiT(d=64, depth=3); d.load_state_dict(torch.load("dit_digits.pt")); d.eval()
for nm, mod in [("U-Net", m), ("DiT", d)]:
    S = sample(mod, 500, y, betas, abar, steps=50, guidance=2.0); r = score(S, y); res.setdefault("arch", {})[nm] = r
    torch.save(S[:40], f"samples_{nm}.pt"); print(nm, {k: round(v, 3) for k, v in r.items()})
json.dump(res, open("eval9.json", "w"), indent=1)
