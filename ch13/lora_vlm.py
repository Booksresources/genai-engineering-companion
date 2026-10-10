import json, math, time, numpy as np, torch, torch.nn as nn
from vlm import *
class LoRA(nn.Module):
    def __init__(s, base, r=8, alpha=16):
        super().__init__(); s.base, s.scale = base, alpha / r
        s.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features)); s.B = nn.Parameter(torch.zeros(base.out_features, r))
    @property
    def weight(s): return s.base.weight + s.scale * s.B @ s.A
    @property
    def bias(s): return s.base.bias
    def forward(s, x): return s.base(x) + (x @ s.A.T @ s.B.T) * s.scale
def load():
    m = TinyVLM(); m.load_state_dict(torch.load("vlm.pt")); return m
OLD = ["count", "tallest", "compare"]; res = {}
def evaluate(m, tag):
    rng = np.random.default_rng(321); r = {t: accuracy(m, rng, t) for t in OLD + ["new"]}; res[tag] = r; print(tag, {k: round(v, 3) for k, v in r.items()})
evaluate(load(), "before (zero-shot)")
# LoRA on the feed-forward layers of the language model; everything else frozen
torch.manual_seed(0); m = load()
for p in m.parameters(): p.requires_grad_(False)
for layer in m.lm.layers:
    layer.linear1, layer.linear2 = LoRA(layer.linear1), LoRA(layer.linear2)
m.tok.weight.requires_grad_(True)                          # let the new word "are" get an embedding
params = [p for p in m.parameters() if p.requires_grad]; res["lora_trainable"] = sum(p.numel() for p in params)
t0 = time.time(); train(m, ["new"], steps=500, lr=2e-3, params=params, seed=1); res["lora_seconds"] = time.time() - t0
evaluate(m, "LoRA on new task only")
torch.manual_seed(0); m = load(); t0 = time.time(); train(m, ["new"], steps=500, lr=1e-3, seed=1); res["full_seconds"] = time.time() - t0
evaluate(m, "full fine-tune on new task only")
torch.manual_seed(0); m = load(); train(m, ["new"] * 3 + OLD, steps=500, lr=1e-3, seed=1)
evaluate(m, "full fine-tune, new + old mixed")
json.dump(res, open("lora_vlm.json", "w"), indent=1)
