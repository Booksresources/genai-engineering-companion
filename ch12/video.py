"""Tiny video diffusion: moving squares (8 frames of 8x8). Per-frame image diffusion vs. a space-time (3D conv) model."""
import sys, math, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/claude/gb/ch9")
from diff9 import schedule, temb
Tn, S = 8, 8
def clips(n, seed=0):
    rng = np.random.default_rng(seed); V = np.full((n, Tn, S, S), -1.0)
    for i in range(n):
        x, y = rng.uniform(0, S - 2, 2); vx, vy = rng.choice([-1, -0.5, 0.5, 1], 2)
        for t in range(Tn):
            xi, yi = int(round(x)), int(round(y)); V[i, t, yi:yi + 2, xi:xi + 2] = 1.0
            x, y = x + vx, y + vy
            if x < 0 or x > S - 2: vx = -vx; x = min(max(x, 0), S - 2)
            if y < 0 or y > S - 2: vy = -vy; y = min(max(y, 0), S - 2)
    return torch.tensor(V, dtype=torch.float32)[:, None]          # (N, 1, T, H, W)
class Block3D(nn.Module):
    def __init__(s, c, e, k=(3, 3, 3)):
        super().__init__(); s.n1 = nn.GroupNorm(8, c); s.c1 = nn.Conv3d(c, c, k, padding=tuple(x // 2 for x in k)); s.e = nn.Linear(e, c)
        s.n2 = nn.GroupNorm(8, c); s.c2 = nn.Conv3d(c, c, k, padding=tuple(x // 2 for x in k))
    def forward(s, x, emb): return x + s.c2(F.silu(s.n2(s.c1(F.silu(s.n1(x))) + s.e(emb)[:, :, None, None, None])))
class Denoiser(nn.Module):
    """temporal=True: 3x3x3 kernels mix information across frames. temporal=False: 1x3x3 kernels see one frame at a time."""
    def __init__(s, temporal, c=32, e=64, depth=4):
        super().__init__(); k = (3, 3, 3) if temporal else (1, 3, 3); s.e = e
        s.inp = nn.Conv3d(1, c, k, padding=tuple(x // 2 for x in k)); s.blocks = nn.ModuleList([Block3D(c, e, k) for _ in range(depth)])
        s.temb = nn.Sequential(nn.Linear(e, e), nn.SiLU(), nn.Linear(e, e)); s.out = nn.Conv3d(c, 1, 1)
    def forward(s, x, t):
        emb = s.temb(temb(t, s.e)); h = s.inp(x)
        for b in s.blocks: h = b(h, emb)
        return s.out(h)
def train(m, V, abar, steps, bs=32, lr=2e-3, seed=0):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.AdamW(m.parameters(), lr=lr); T = len(abar); log = []
    for s_ in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr * min(1, (s_ + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s_ / steps))
        x0 = V[torch.randint(0, len(V), (bs,), generator=g)]; t = torch.randint(0, T, (bs,), generator=g); e = torch.randn(x0.shape, generator=g)
        a = abar[t].view(-1, 1, 1, 1, 1); loss = F.mse_loss(m(a.sqrt() * x0 + (1 - a).sqrt() * e, t), e)
        opt.zero_grad(); loss.backward(); opt.step()
        if s_ % 100 == 0: log.append((s_, loss.item()))
    return log
@torch.no_grad()
def sample(m, n, abar, steps=20, first_frame=None, seed=0):
    """DDIM sampling; with first_frame, frame 0 is clamped to the given image at every step (image-to-video)."""
    g = torch.Generator().manual_seed(seed); T = len(abar); x = torch.randn((n, 1, Tn, S, S), generator=g)
    ts = torch.linspace(T - 1, 0, steps).long().tolist()
    for i, t in enumerate(ts):
        if first_frame is not None:
            x[:, :, 0] = abar[t].sqrt() * first_frame + (1 - abar[t]).sqrt() * torch.randn(first_frame.shape, generator=g)
        e = m(x, torch.full((n,), t)); a_t = abar[t]; a_p = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x0 = ((x - (1 - a_t).sqrt() * e) / a_t.sqrt()).clamp(-1, 1); x = a_p.sqrt() * x0 + (1 - a_p).sqrt() * e
    if first_frame is not None: x[:, :, 0] = first_frame
    return x.clamp(-1, 1)
def analyze(V):
    """Per clip: is there exactly one square in every frame, how far does it jump between frames, and is motion steady?"""
    V = V[:, 0].numpy(); out = {"one_object": [], "jump": [], "steady": []}
    for clip in V:
        pos, ok = [], True
        for f in clip:
            on = f > 0.0; n_on = on.sum()
            if not (2 <= n_on <= 6): ok = False
            ys, xs = np.nonzero(on); pos.append((xs.mean() if len(xs) else np.nan, ys.mean() if len(ys) else np.nan))
        pos = np.array(pos); d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        out["one_object"].append(ok); out["jump"].append(np.nanmean(d) if np.isfinite(d).any() else np.nan)
        out["steady"].append(bool(ok and np.nanmax(d) <= 1.6))
    return {"one_object_every_frame": float(np.mean(out["one_object"])), "mean_jump_px": float(np.nanmean(out["jump"])), "smooth_motion": float(np.mean(out["steady"]))}
if __name__ == "__main__":
    which = sys.argv[1]; torch.manual_seed(0); V = clips(3000); betas, abar = schedule("cosine", 200); t0 = time.time()
    m = Denoiser(temporal=(which == "video")); log = train(m, V, abar, steps=int(sys.argv[2]))
    torch.save(m.state_dict(), f"{which}.pt"); print(which, "params", sum(p.numel() for p in m.parameters()), log[-1], f"{time.time()-t0:.0f}s")
    json.dump({"log": log, "seconds": time.time() - t0, "params": sum(p.numel() for p in m.parameters())}, open(f"{which}_train.json", "w"))
