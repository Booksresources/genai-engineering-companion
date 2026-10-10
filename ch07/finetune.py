import sys, json, math, re, time, random, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/claude"); sys.path.insert(0, "/home/claude/ch15")
from mini_gpt import GPT, GPTConfig
from nf4 import quantize_blockwise, NF4
D = json.load(open("/home/claude/gb/ch7/brand_data.json")); TRAIN, TEST, CLOSERS = D["train"], D["test"], D["closers"]
text = open("/home/claude/tiny.txt").read(); chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
def enc(s): return [stoi[c] for c in s]
def prompt_for(name, feat): return f"P: {name}; f: {feat}\nC: "
def load_base():
    m = GPT(GPTConfig(vocab_size=65)); m.load_state_dict(torch.load("/home/claude/ckpt.pt", weights_only=False)["model"]); return m

class LoRALinear(nn.Module):
    """Frozen base weight W plus a trainable low-rank update (alpha/r) * B @ A."""
    def __init__(self, base, r=8, alpha=16):
        super().__init__(); self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r)); base.weight.requires_grad_(False)
    def forward(self, x): return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
def add_lora(m, r=8):
    for p in m.parameters(): p.requires_grad_(False)
    for blk in m.blocks:
        for parent, names in [(blk.attn, ["wq", "wk", "wv", "wo"]), (blk.ffn, ["w1", "w2", "w3"])]:
            for n in names: setattr(parent, n, LoRALinear(getattr(parent, n), r))
    return [p for p in m.parameters() if p.requires_grad]
def quantize_base_nf4(m):                      # QLoRA-style: store frozen linear weights in 4-bit NF4 (simulated)
    for blk in m.blocks:
        for lin in [blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.attn.wo, blk.ffn.w1, blk.ffn.w2, blk.ffn.w3]:
            lin.weight.data = quantize_blockwise(lin.weight.data, NF4)

def batch(rows, bs, rng):
    xs, ys = [], []
    for r in rng.sample(rows, bs):
        p, a = enc(prompt_for(r["product"], r["feature"])), enc(r["copy"] + "\n")
        ids = (p + a)[:129]; lab = ([-100] * len(p) + a)[:129]        # loss only on the copy
        ids += [0] * (129 - len(ids)); lab += [-100] * (129 - len(lab))
        xs.append(ids[:-1]); ys.append(lab[1:])
    return torch.tensor(xs), torch.tensor(ys)

def train(m, params, steps, lr, seed=0):
    rng = random.Random(seed); opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0); hist = []
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr * min(1, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * s / steps))
        x, y = batch(TRAIN, 32, rng); logits, _ = m(x)
        loss = F.cross_entropy(logits.reshape(-1, 65), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if s % 25 == 0: hist.append((s, loss.item()))
    return hist

@torch.no_grad()
def generate_copy(m, rows, few_shot=""):
    m.eval(); outs = []
    for r in rows:
        ids = torch.tensor([enc(few_shot + prompt_for(r["product"], r["feature"]))])[:, -128:]
        for _ in range(110):
            logits, _ = m(ids[:, -128:]); nxt = logits[0, -1].argmax().item()
            if chars[nxt] == "\n": break
            ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
        full = "".join(chars[i] for i in ids[0].tolist()); outs.append(full.split("\nC: ")[-1])
    return outs

def brand_check(r, out):
    feat = r["feature"][0].upper() + r["feature"][1:]
    pat = rf"^Meet the {re.escape(r['product'])}\. {re.escape(feat)}\. ({'|'.join(map(re.escape, CLOSERS))}) Northwind\.$"
    return {"pass": bool(re.match(pat, out)), "name": f"Meet the {r['product']}." in out, "feature": feat in out,
            "signoff": out.endswith("Northwind."), "no_exclaim": "!" not in out}

@torch.no_grad()
def shakespeare_loss(m):
    m.eval(); data = torch.tensor(enc(text[int(0.9 * len(text)):])); g = torch.Generator().manual_seed(1); tot = 0
    for _ in range(10):
        ix = torch.randint(0, len(data) - 129, (32,), generator=g)
        tot += m(torch.stack([data[i:i + 128] for i in ix]), torch.stack([data[i + 1:i + 129] for i in ix]))[1].item()
    return tot / 10

if __name__ == "__main__":
    mode = sys.argv[1]; t0 = time.time(); torch.manual_seed(0); m = load_base(); res = {"mode": mode}
    total = sum(p.numel() for p in m.parameters())
    if mode == "prompt_base":
        ex = TRAIN[0]; shots = "".join(prompt_for(r["product"], r["feature"]) + r["copy"] + "\n" for r in TRAIN[:1])
        outs = generate_copy(m, TEST, few_shot=shots); res["trainable"] = 0
    else:
        if mode == "qlora": quantize_base_nf4(m)
        params = list(m.parameters()) if mode == "full" else add_lora(m, r=8)
        res["trainable"] = sum(p.numel() for p in params)
        res["hist"] = train(m, params, steps=250, lr=1e-3 if mode == "full" else 3e-3)
        outs = generate_copy(m, TEST)
        if mode == "lora": torch.save({k: v for k, v in m.state_dict().items() if ".A" in k or ".B" in k}, "adapter.pt")
    checks = [brand_check(r, o) for r, o in zip(TEST, outs)]
    res.update({"total_params": total, "pass_rate": sum(c["pass"] for c in checks) / len(checks),
                "parts": {k: sum(c[k] for c in checks) / len(checks) for k in ["name", "feature", "signoff", "no_exclaim"]},
                "examples": outs[:3], "shakespeare_loss": shakespeare_loss(m), "seconds": time.time() - t0})
    print(json.dumps({k: v for k, v in res.items() if k != "hist"}, indent=1))
    try: allres = json.load(open("ft_results.json"))
    except FileNotFoundError: allres = {}
    allres[mode] = res; json.dump(allres, open("ft_results.json", "w"), indent=1)
