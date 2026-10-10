"""A tiny vision-language model for chart question answering: CNN encoder -> projector -> Transformer LM."""
import sys, json, math, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
WORDS = ["<pad>", "how", "many", "bars", "?", "which", "bar", "is", "tallest", "taller", "than", "1", "2", "3", "4", "5", "yes", "no", "0", "are"]
W = {w: i for i, w in enumerate(WORDS)}
def chart(heights, size=16):
    img = np.full((size, size), -1.0); n = len(heights); w = 2; gap = (size - n * w) // (n + 1)
    for i, h in enumerate(heights):
        x0 = gap + i * (w + gap); img[size - h:, x0:x0 + w] = 1.0
    return img
def example(rng, task):
    n = int(rng.integers(3, 6)); hs = [int(v) for v in rng.choice(np.arange(3, 15), n, replace=False)]
    if task == "count": q, a = "how many bars ?", str(n)
    elif task == "tallest": q, a = "which bar is tallest ?", str(int(np.argmax(hs)) + 1)
    elif task == "compare":
        i, j = rng.choice(n, 2, replace=False); q, a = f"is bar {i+1} taller than bar {j+1} ?", "yes" if hs[i] > hs[j] else "no"
    else:   # new task for the LoRA section
        q, a = "how many bars are taller than bar 1 ?", str(sum(h > hs[0] for h in hs[1:]))
    return chart(hs), q, a
def batch(rng, n, tasks):
    imgs, qs, ans = [], [], []
    for _ in range(n):
        im, q, a = example(rng, tasks[int(rng.integers(0, len(tasks)))]); imgs.append(im); qs.append([W[w] for w in q.split()]); ans.append(W[a])
    L = 10; qt = torch.tensor([q + [0] * (L - len(q)) for q in qs])
    return torch.tensor(np.array(imgs), dtype=torch.float32)[:, None], qt, torch.tensor(ans)
class TinyVLM(nn.Module):
    def __init__(s, d=96, layers=3, use_image=True):
        super().__init__(); s.use_image = use_image
        s.vision = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.GELU())  # 16x16 -> 4x4 grid
        s.projector = nn.Sequential(nn.Linear(64, d), nn.GELU(), nn.Linear(d, d))       # visual features -> LM embedding space
        s.vpos = nn.Parameter(torch.randn(1, 16, d) * 0.02)
        s.tok = nn.Embedding(len(WORDS), d); s.pos = nn.Parameter(torch.randn(1, 11, d) * 0.02)
        s.lm = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, norm_first=True), layers)
        s.head = nn.Linear(d, len(WORDS))
    def forward(s, img, q):
        B = q.shape[0]; ans_slot = torch.zeros(B, 1, dtype=torch.long)
        t = s.tok(torch.cat([q, ans_slot], 1)) + s.pos
        if s.use_image:
            v = s.projector(s.vision(img).flatten(2).transpose(1, 2)) + s.vpos             # 16 visual tokens
            h = torch.cat([v, t], 1)
        else: h = t
        pad = torch.cat([torch.zeros(B, 16 if s.use_image else 0, dtype=torch.bool), torch.cat([q, ans_slot], 1) == 0], 1); pad[:, -1] = False
        return s.head(s.lm(h, src_key_padding_mask=pad)[:, -1])                           # answer predicted at the final slot
def accuracy(m, rng, task, n=500):
    m.eval()
    with torch.no_grad(): im, q, a = batch(rng, n, [task]); r = (m(im, q).argmax(-1) == a).float().mean().item()
    m.train(); return r
def train(m, tasks, steps, lr=2e-3, params=None, seed=0):
    rng = np.random.default_rng(seed); opt = torch.optim.AdamW(params or m.parameters(), lr=lr); log = []
    for s_ in range(steps):
        for g in opt.param_groups: g["lr"] = lr * min(1, (s_ + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s_ / steps))
        im, q, a = batch(rng, 64, tasks); loss = F.cross_entropy(m(im, q), a)
        opt.zero_grad(); loss.backward(); opt.step()
        if s_ % 100 == 0: log.append((s_, loss.item()))
    return log
if __name__ == "__main__":
    which = sys.argv[1]; torch.manual_seed(0); t0 = time.time(); TASKS = ["count", "tallest", "compare"]
    m = TinyVLM(use_image=(which == "vlm")); log = train(m, TASKS, steps=int(sys.argv[2]))
    rng = np.random.default_rng(123); acc = {t: accuracy(m, rng, t) for t in TASKS}
    torch.save(m.state_dict(), f"{which}.pt")
    out = {"acc": acc, "log": log, "seconds": time.time() - t0, "params": sum(p.numel() for p in m.parameters())}
    json.dump(out, open(f"{which}.json", "w"), indent=1); print(which, acc, f"{out['seconds']:.0f}s", out["params"])
