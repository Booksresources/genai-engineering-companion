import sys, json, time, copy, io, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/claude"); sys.path.insert(0, "/home/claude/ch18")
from mini_gpt import GPT, GPTConfig
from kvgpt import generate_cached, generate_nocache
torch.manual_seed(0); res = {}
text = open("/home/claude/tiny.txt").read(); chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
m = GPT(GPTConfig(vocab_size=65)); m.load_state_dict(torch.load("/home/claude/ckpt.pt", weights_only=False)["model"]); m.eval()
def tm(f, reps=3):
    best = 1e9
    for _ in range(reps): t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
    return best
# 1) prefill vs decode: one forward over a prompt of L tokens vs L single-token decode steps
res["prefill_decode"] = {}
with torch.no_grad():
    for L in [16, 64, 120]:
        ids = torch.randint(0, 65, (1, L))
        pre = tm(lambda: m(ids)); dec = tm(lambda: [m(ids[:, :i + 1]) for i in range(L)], reps=1)
        res["prefill_decode"][L] = {"prefill_s": pre, "token_by_token_s": dec}
print("prefill vs decode:", res["prefill_decode"])
# 2) KV cache
prompt = torch.tensor([[stoi[c] for c in "ROMEO:\n"]])
with torch.no_grad():
    res["kv"] = {}
    for n in [60, 120]:
        res["kv"][n] = {"no_cache_s": tm(lambda: generate_nocache(m, prompt, n), 1), "cache_s": tm(lambda: generate_cached(m, prompt, n), 1)}
print("kv:", res["kv"])
# 3) batching: decode-step throughput by batch size (one forward per step over 64-token contexts)
res["batch"] = {}
with torch.no_grad():
    for B in [1, 4, 16, 64]:
        x = torch.randint(0, 65, (B, 64)); t = tm(lambda: m(x)); res["batch"][B] = {"step_s": t, "tokens_per_s": B / t}
print("batch:", res["batch"])
# 4) quantization of the tiny GPT: INT8 dynamic (real kernels) and simulated INT4 group-wise
data = torch.tensor([stoi[c] for c in text[int(0.9 * len(text)):]])
@torch.no_grad()
def val_loss(model):
    g = torch.Generator().manual_seed(1); tot = 0
    for _ in range(10):
        ix = torch.randint(0, len(data) - 129, (16,), generator=g)
        tot += model(torch.stack([data[i:i + 128] for i in ix]), torch.stack([data[i + 1:i + 129] for i in ix]))[1].item()
    return tot / 10
def size_mb(model): b = io.BytesIO(); torch.save(model.state_dict(), b); return b.getbuffer().nbytes / 1e6
q8 = torch.ao.quantization.quantize_dynamic(copy.deepcopy(m), {torch.nn.Linear}, dtype=torch.qint8)
def int4_sim(model, group=32):
    mm = copy.deepcopy(model)
    for mod in mm.modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight.data; g = w.view(-1, group); s = g.abs().amax(1, keepdim=True) / 7
            mod.weight.data = (torch.clamp(torch.round(g / s), -8, 7) * s).view_as(w)
    return mm
q4 = int4_sim(m); x = torch.randint(0, 65, (16, 128))
res["quant"] = {"fp32": {"loss": val_loss(m), "mb": size_mb(m), "fwd_s": tm(lambda: m(x))},
                "int8 dynamic": {"loss": val_loss(q8), "mb": size_mb(q8), "fwd_s": tm(lambda: q8(x))},
                "int4 group-32 (simulated)": {"loss": val_loss(q4), "mb": size_mb(m) * 4.5 / 32}}
print("quant:", res["quant"])
json.dump(res, open("infer.json", "w"), indent=1)
