import sys, json, time, copy, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/gb/ch9"); sys.path.insert(0, "/home/claude"); sys.path.insert(0, "/home/claude/ch18")
from diff9 import UNet, schedule, digits
from metrics9 import score
res = {}
betas, abar = schedule("cosine", 200); m = UNet(16, 10); m.load_state_dict(torch.load("/home/claude/gb/ch9/unet_digits.pt")); m.eval()
y = torch.arange(500) % 10
@torch.no_grad()
def ddim(model, steps, reuse_every=1, seed=0):
    g = torch.Generator().manual_seed(seed); x = torch.randn((500, 1, 8, 8), generator=g); ts = torch.linspace(199, 0, steps).long().tolist(); e = None; calls = 0
    for i, t in enumerate(ts):
        if e is None or i % reuse_every == 0: e = model(x, torch.full((500,), t), y); calls += 1      # reuse the last noise estimate in between
        a_t = abar[t]; a_p = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x0 = ((x - (1 - a_t).sqrt() * e) / a_t.sqrt()).clamp(-1, 1); x = a_p.sqrt() * x0 + (1 - a_p).sqrt() * e
    return x.clamp(-1, 1), calls
def run(name, model, steps, reuse=1):
    t = time.perf_counter(); S, calls = ddim(model, steps, reuse); dt = time.perf_counter() - t; r = score(S, y)
    res.setdefault("diffusion", {})[name] = {"fd": r["fd"], "class_match": r["class_match"], "seconds": dt, "network_calls": calls}; print(name, res["diffusion"][name])
run("50 steps", m, 50); run("20 steps", m, 20); run("10 steps", m, 10); run("20 steps, reuse every 2nd", m, 20, 2)
m8 = copy.deepcopy(m)
for mod in m8.modules():
    if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d)):
        w = mod.weight.data; s = w.abs().amax() / 127; mod.weight.data = torch.round(w / s) * s
run("20 steps, INT8 weights (simulated)", m8, 20)
# ---- halving latency of a Muse copy request on the tiny GPT
from mini_gpt import GPT, GPTConfig
from kvgpt import generate_cached, generate_nocache
gm = GPT(GPTConfig(vocab_size=65)); gm.load_state_dict(torch.load("/home/claude/ckpt.pt", weights_only=False)["model"]); gm.eval()
q8 = torch.ao.quantization.quantize_dynamic(copy.deepcopy(gm), {torch.nn.Linear}, dtype=torch.qint8)
prompt = torch.randint(0, 65, (1, 20))
def t_(f, reps=3):
    b = 1e9
    for _ in range(reps): t = time.perf_counter(); f(); b = min(b, time.perf_counter() - t)
    return b
with torch.no_grad():
    steps = [("baseline: no KV cache, 100 tokens", lambda: generate_nocache(gm, prompt, 100)),
             ("+ KV cache", lambda: generate_cached(gm, prompt, 100)),
             ("+ INT8 weights", lambda: generate_cached(q8, prompt, 100)),
             ("+ max_tokens 60 (shorter output)", lambda: generate_cached(q8, prompt, 60))]
    res["halving"] = {}
    for name, f in steps: res["halving"][name] = t_(f); print(name, round(res["halving"][name], 3))
    b4 = prompt.repeat(4, 1); per = t_(lambda: generate_cached(q8, b4, 60)) / 4
    res["halving"]["+ batch 4 requests (time per request)"] = per; print("batch4 per request", round(per, 3))
json.dump(res, open("infer2.json", "w"), indent=1)
