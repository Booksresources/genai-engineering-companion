"""Diffusion from scratch: data, schedules, a small U-Net, a tiny DiT, DDPM and DDIM samplers, CFG."""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits

# ---------------- data ----------------
def digits():
    d = load_digits(); x = torch.tensor(d.data, dtype=torch.float32).view(-1, 1, 8, 8) / 16 * 2 - 1
    return x, torch.tensor(d.target)
def faces(n=3000, seed=0, size=16):
    """Synthetic cartoon faces with labels 0 = frown, 1 = neutral, 2 = smile (an offline stand-in for face datasets)."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:size, 0:size] + 0.5; imgs, labs = [], []
    for _ in range(n):
        cx, cy, r = size / 2 + rng.uniform(-1, 1), size / 2 + rng.uniform(-1, 1), rng.uniform(5.5, 7.2)
        img = np.full((size, size), -1.0)
        img[(xx - cx) ** 2 + ((yy - cy) / 1.1) ** 2 < r ** 2] = rng.uniform(0.2, 0.8)          # head
        ey, ex = cy - r * 0.3, r * 0.38; er = rng.uniform(0.8, 1.3)
        for sx in (-1, 1): img[(xx - cx - sx * ex) ** 2 + (yy - ey) ** 2 < er ** 2] = -0.8        # eyes
        lab = rng.integers(0, 3); my = cy + r * 0.42; w = r * 0.55; curv = {0: -0.6, 1: 0.0, 2: 0.6}[lab]
        for t in np.linspace(-1, 1, 30):                                                    # mouth curve
            px, py = cx + t * w, my + curv * (1 - t * t) * 1.6
            img[(np.abs(xx - px) < 0.7) & (np.abs(yy - py) < 0.7)] = -0.9
        if rng.random() < 0.3: img[(np.abs(yy - ey) < 0.6) & (np.abs(xx - cx) < ex + er)] = -0.9  # glasses bar
        imgs.append(img); labs.append(lab)
    return torch.tensor(np.array(imgs), dtype=torch.float32)[:, None], torch.tensor(labs)

# ---------------- schedules ----------------
def schedule(kind, T):
    if kind == "linear":
        betas = torch.linspace(1e-4, 0.02 * 1000 / T, T)
    else:                                                   # cosine (Nichol & Dhariwal 2021)
        s = 0.008; t = torch.linspace(0, T, T + 1) / T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2; abar = f / f[0]
        betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    return betas, torch.cumprod(1 - betas, 0)

def temb(t, dim):
    f = torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2))
    a = t.float()[:, None] * f[None]; return torch.cat([a.sin(), a.cos()], 1)

# ---------------- models ----------------
class ResBlock(nn.Module):
    def __init__(s, c_in, c_out, e):
        super().__init__(); s.n1 = nn.GroupNorm(8, c_in); s.c1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        s.n2 = nn.GroupNorm(8, c_out); s.c2 = nn.Conv2d(c_out, c_out, 3, padding=1); s.e = nn.Linear(e, c_out)
        s.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(s, x, emb):
        h = s.c1(F.silu(s.n1(x))) + s.e(emb)[:, :, None, None]
        return s.skip(x) + s.c2(F.silu(s.n2(h)))
class UNet(nn.Module):
    """A small U-Net: two resolutions, skip connections, time and class embeddings."""
    def __init__(s, ch=32, n_classes=10, e=128):
        super().__init__(); s.e = e; s.n_classes = n_classes
        s.temb = nn.Sequential(nn.Linear(e, e), nn.SiLU(), nn.Linear(e, e)); s.cemb = nn.Embedding(n_classes + 1, e)   # last = "no label"
        s.inp = nn.Conv2d(1, ch, 3, padding=1)
        s.d1 = ResBlock(ch, ch, e); s.down = nn.Conv2d(ch, 2 * ch, 4, 2, 1)
        s.mid = ResBlock(2 * ch, 2 * ch, e); s.up = nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1)
        s.u1 = ResBlock(2 * ch, ch, e); s.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))
    def forward(s, x, t, y):
        emb = s.temb(temb(t, s.e)) + s.cemb(y)
        h1 = s.d1(s.inp(x), emb); h = s.mid(s.down(h1), emb); h = s.up(h)
        return s.out(s.u1(torch.cat([h, h1], 1), emb))
class DiT(nn.Module):
    """A tiny diffusion transformer: patchify, Transformer blocks with adaptive layer norm, unpatchify."""
    def __init__(s, size=8, patch=2, d=96, depth=4, heads=4, n_classes=10):
        super().__init__(); s.p, s.size, s.d = patch, size, d; n = (size // patch) ** 2
        s.embed = nn.Linear(patch * patch, d); s.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        s.temb = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)); s.cemb = nn.Embedding(n_classes + 1, d)
        s.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True, activation="gelu") for _ in range(depth)])
        s.ada = nn.ModuleList([nn.Linear(d, 2 * d) for _ in range(depth)])        # scale and shift from the condition
        s.out = nn.Linear(d, patch * patch)
    def forward(s, x, t, y):
        B, p, S = x.shape[0], s.p, s.size
        tok = x.unfold(2, p, p).unfold(3, p, p).reshape(B, 1, -1, p * p)[:, 0]
        h = s.embed(tok) + s.pos; c = s.temb(temb(t, s.d)) + s.cemb(y)
        for blk, ada in zip(s.blocks, s.ada):
            sc, sh = ada(c).chunk(2, -1); h = blk(h * (1 + sc[:, None]) + sh[:, None])
        o = s.out(h).view(B, S // p, S // p, p, p).permute(0, 1, 3, 2, 4).reshape(B, 1, S, S)
        return o

# ---------------- training and sampling ----------------
def train(model, X, Y, abar, steps=3000, bs=128, lr=2e-3, p_uncond=0.15, seed=0, log=None):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.AdamW(model.parameters(), lr=lr); T = len(abar)
    for s in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr * min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps))
        i = torch.randint(0, len(X), (bs,), generator=g); x0, y = X[i], Y[i].clone()
        y[torch.rand(bs, generator=g) < p_uncond] = model.n_classes if hasattr(model, "n_classes") else 10   # label dropout for CFG
        t = torch.randint(0, T, (bs,), generator=g); e = torch.randn(x0.shape, generator=g)
        a = abar[t].view(-1, 1, 1, 1); xt = a.sqrt() * x0 + (1 - a).sqrt() * e
        loss = F.mse_loss(model(xt, t, y), e)
        opt.zero_grad(); loss.backward(); opt.step()
        if log is not None and s % 100 == 0: log.append((s, loss.item()))
    return model

@torch.no_grad()
def sample(model, n, y, betas, abar, steps=None, guidance=1.0, eta=0.0, shape=(1, 8, 8), seed=0):
    """DDPM ancestral sampling (steps=None) or DDIM with `steps` evenly spaced steps; classifier-free guidance."""
    g = torch.Generator().manual_seed(seed); T = len(abar); x = torch.randn((n,) + shape, generator=g)
    null = torch.full_like(y, getattr(model, "n_classes", 10))
    def eps(x, t):
        tt = torch.full((n,), t, dtype=torch.long)
        if guidance == 1.0: return model(x, tt, y)
        e_c, e_u = model(x, tt, y), model(x, tt, null)
        return e_u + guidance * (e_c - e_u)                                   # classifier-free guidance
    if steps is None:
        for t in reversed(range(T)):
            e = eps(x, t); x = (x - betas[t] / (1 - abar[t]).sqrt() * e) / (1 - betas[t]).sqrt()
            if t > 0: x = x + betas[t].sqrt() * torch.randn(x.shape, generator=g)
        return x.clamp(-1, 1)
    ts = torch.linspace(T - 1, 0, steps).long().tolist()
    for i, t in enumerate(ts):
        e = eps(x, t); a_t = abar[t]; a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x0 = ((x - (1 - a_t).sqrt() * e) / a_t.sqrt()).clamp(-1, 1)          # predicted clean image
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e                       # deterministic DDIM step (eta = 0)
    return x.clamp(-1, 1)
