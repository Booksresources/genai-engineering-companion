"""A miniature text-to-image system: captioned synthetic faces, a word-embedding text encoder,
a U-Net conditioned on the text embedding, classifier-free guidance, inpainting, and image-to-image."""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/claude/gb/ch9")
from diff9 import ResBlock, temb, schedule
EXPR = ["frowning", "neutral", "smiling"]
VOCAB = ["<pad>", "a", "face", "frowning", "neutral", "smiling", "with", "glasses", "without", "on", "white", "grey", "background", "hat", "sks"]
W = {w: i for i, w in enumerate(VOCAB)}

def render(expr, glasses, bg_grey, hat=False, rng=None, size=16):
    rng = rng or np.random.default_rng(); yy, xx = np.mgrid[0:size, 0:size] + 0.5
    cx, cy, r = size / 2 + rng.uniform(-0.8, 0.8), size / 2 + rng.uniform(-0.5, 1.0), rng.uniform(5.2, 6.4)
    img = np.full((size, size), -0.35 if bg_grey else -1.0)
    img[(xx - cx) ** 2 + ((yy - cy) / 1.1) ** 2 < r ** 2] = rng.uniform(0.45, 0.85)
    ey, ex = cy - r * 0.3, r * 0.38; er = rng.uniform(0.8, 1.2)
    for sx in (-1, 1): img[(xx - cx - sx * ex) ** 2 + (yy - ey) ** 2 < er ** 2] = -0.85
    my, w = cy + r * 0.42, r * 0.55; curv = {0: -0.6, 1: 0.0, 2: 0.6}[expr]
    for t in np.linspace(-1, 1, 30):
        px, py = cx + t * w, my + curv * (1 - t * t) * 1.6
        img[(np.abs(xx - px) < 0.7) & (np.abs(yy - py) < 0.7)] = -0.9
    if glasses: img[(np.abs(yy - ey) < 0.6) & (np.abs(xx - cx) < ex + er)] = -0.9
    if hat: img[(yy < cy - r * 0.75) & (yy > cy - r * 1.35) & (np.abs(xx - cx) < r * 0.9)] = 0.95   # bright hat band
    return img

def caption(expr, glasses, bg_grey, hat=False, concept=None):
    words = ["a", concept] if concept else ["a"]
    words += [EXPR[expr], "face", "with" if glasses else "without", "glasses", "on", "grey" if bg_grey else "white", "background"]
    return " ".join(words)

def dataset(n, seed=0):
    rng = np.random.default_rng(seed); X, C, A = [], [], []
    for _ in range(n):
        e, g, b = rng.integers(0, 3), rng.random() < 0.5, rng.random() < 0.5
        X.append(render(e, g, b, rng=rng)); C.append(caption(e, g, b)); A.append((e, g, b))
    return torch.tensor(np.array(X), dtype=torch.float32)[:, None], C, A

def tokenize(caps, L=10):
    ids = [[W[w] for w in c.split()][:L] for c in caps]
    return torch.tensor([x + [0] * (L - len(x)) for x in ids])

class TextEncoder(nn.Module):
    """Word embeddings + a small Transformer layer + mean pooling (stands in for CLIP / T5)."""
    def __init__(s, d=64):
        super().__init__(); s.emb = nn.Embedding(len(VOCAB), d); s.pos = nn.Parameter(torch.randn(1, 10, d) * 0.02)
        s.layer = nn.TransformerEncoderLayer(d, 4, 128, batch_first=True)
    def forward(s, ids):
        h = s.layer(s.emb(ids) + s.pos, src_key_padding_mask=(ids == 0) & (ids.sum(1, keepdim=True) > 0))
        mask = (ids != 0).float().unsqueeze(-1); return (h * mask).sum(1) / mask.sum(1).clamp(min=1)

class TextUNet(nn.Module):
    def __init__(s, ch=16, e=128):
        super().__init__(); s.e = e; s.text = TextEncoder(); s.tproj = nn.Linear(64, e)
        s.temb = nn.Sequential(nn.Linear(e, e), nn.SiLU(), nn.Linear(e, e))
        s.inp = nn.Conv2d(1, ch, 3, padding=1); s.d1 = ResBlock(ch, ch, e); s.down = nn.Conv2d(ch, 2 * ch, 4, 2, 1)
        s.mid = ResBlock(2 * ch, 2 * ch, e); s.up = nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1); s.u1 = ResBlock(2 * ch, ch, e)
        s.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))
    def forward(s, x, t, ids):
        emb = s.temb(temb(t, s.e)) + s.tproj(s.text(ids))
        h1 = s.d1(s.inp(x), emb); h = s.up(s.mid(s.down(h1), emb)); return s.out(s.u1(torch.cat([h, h1], 1), emb))

def train(model, X, ids, abar, steps, bs=64, lr=2e-3, p_uncond=0.15, seed=0, params=None, log=None):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.AdamW(params or model.parameters(), lr=lr); T = len(abar)
    for s in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr * min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps))
        i = torch.randint(0, len(X), (bs,), generator=g); x0, c = X[i], ids[i].clone()
        c[torch.rand(bs, generator=g) < p_uncond] = 0                         # empty prompt = unconditional
        t = torch.randint(0, T, (bs,), generator=g); e = torch.randn(x0.shape, generator=g)
        a = abar[t].view(-1, 1, 1, 1); loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * e, t, c), e)
        opt.zero_grad(); loss.backward(); opt.step()
        if log is not None and s % 100 == 0: log.append((s, loss.item()))

@torch.no_grad()
def generate(model, prompts, abar, steps=20, guidance=3.0, negative=None, seed=0, init=None, strength=1.0, mask=None, known=None):
    """DDIM text-to-image with CFG; optional image-to-image (init + strength) and inpainting (mask: 1 = regenerate)."""
    n = len(prompts); g = torch.Generator().manual_seed(seed); T = len(abar)
    cond = tokenize(prompts); uncond = tokenize(negative) if negative else torch.zeros_like(cond)
    ts = torch.linspace(int((T - 1) * strength), 0, steps).long().tolist()
    noise = torch.randn((n, 1, 16, 16), generator=g)
    x = noise if init is None else abar[ts[0]].sqrt() * init + (1 - abar[ts[0]]).sqrt() * noise
    for i, t in enumerate(ts):
        if mask is not None:                                                   # keep known pixels, re-noised to step t
            x = mask * x + (1 - mask) * (abar[t].sqrt() * known + (1 - abar[t]).sqrt() * torch.randn(x.shape, generator=g))
        tt = torch.full((n,), t); e_c, e_u = model(x, tt, cond), model(x, tt, uncond); e = e_u + guidance * (e_c - e_u)
        a_t = abar[t]; a_p = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x0 = ((x - (1 - a_t).sqrt() * e) / a_t.sqrt()).clamp(-1, 1); x = a_p.sqrt() * x0 + (1 - a_p).sqrt() * e
    if mask is not None: x = mask * x + (1 - mask) * known
    return x.clamp(-1, 1)
