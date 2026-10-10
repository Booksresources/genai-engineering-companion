import sys, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import make_moons
sys.path.insert(0, "/home/claude")
out = {}
# ---- 3.1 temperature on a real next-token distribution
from mini_gpt import GPT, GPTConfig
text = open("/home/claude/tiny.txt").read(); chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
m = GPT(GPTConfig(vocab_size=65)); m.load_state_dict(torch.load("/home/claude/ckpt.pt", weights_only=False)["model"]); m.eval()
with torch.no_grad(): logits = m(torch.tensor([[stoi[c] for c in "To be, or not to b"]]))[0][0, -1]
out["temp"] = {}
for T in [0.5, 1.0, 1.5]:
    p = F.softmax(logits / T, -1); top = torch.topk(p, 6)
    ent = -(p * p.clamp_min(1e-12).log2()).sum().item()
    out["temp"][T] = {"top": [(chars[i], float(v)) for v, i in zip(top.values, top.indices)], "entropy_bits": ent}
p = F.softmax(logits, -1); torch.manual_seed(0); draws = torch.multinomial(p, 2000, replacement=True)
top = torch.topk(p, 6).indices
out["sampling_check"] = [(chars[i], float(p[i]), float((draws == i).float().mean())) for i in top]
print(out["temp"][1.0]["top"][:3], {T: round(v["entropy_bits"], 2) for T, v in out["temp"].items()})

# ---- 3.3 GAN on 8 Gaussians
torch.manual_seed(0)
centers = torch.tensor([[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)]) * 2
def ring(n): return centers[torch.randint(0, 8, (n,))] + 0.08 * torch.randn(n, 2)
G = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
D = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
oG = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), 1e-3, betas=(0.5, 0.999))
snaps = {}; zfix = torch.randn(1000, 4)
def modes(s): return int(sum(((s - c).norm(dim=1) < 0.3).sum() > 20 for c in centers))
for step in range(6001):
    if step in (0, 500, 2000, 6000):
        with torch.no_grad(): s = G(zfix); snaps[step] = {"pts": s.tolist(), "modes": modes(s)}
    real = ring(256); fake = G(torch.randn(256, 4))
    lD = F.binary_cross_entropy_with_logits(D(real), torch.ones(256, 1)) + F.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(256, 1))
    oD.zero_grad(); lD.backward(); oD.step()
    lG = F.binary_cross_entropy_with_logits(D(fake), torch.ones(256, 1)); oG.zero_grad(); lG.backward(); oG.step()
out["gan2d"] = {"snaps": snaps, "centers": centers.tolist()}
print("GAN modes covered:", {k: v["modes"] for k, v in snaps.items()})

# ---- 3.5 diffusion forward process and flow matching on two moons
X, _ = make_moons(4000, noise=0.06, random_state=0); X = torch.tensor((X - X.mean(0)) / X.std(0), dtype=torch.float32)
T = 100; betas = torch.linspace(1e-4, 0.05, T); abar = torch.cumprod(1 - betas, 0)
fwd = {}
for t in [0, 10, 30, 99]:
    e = torch.randn_like(X[:800]); fwd[t] = (abar[t].sqrt() * X[:800] + (1 - abar[t]).sqrt() * e).tolist()
out["forward"] = fwd
torch.manual_seed(0)
v = nn.Sequential(nn.Linear(3, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))
opt = torch.optim.Adam(v.parameters(), 1e-3)
for step in range(5000):
    x1 = X[torch.randint(0, len(X), (512,))]; x0 = torch.randn_like(x1); t = torch.rand(512, 1)
    xt = (1 - t) * x0 + t * x1                         # straight-line path from noise to data
    loss = F.mse_loss(v(torch.cat([xt, t], 1)), x1 - x0)  # learn the constant velocity of that path
    opt.zero_grad(); loss.backward(); opt.step()
def sample(n_steps, n=800, keep=False):
    torch.manual_seed(1); x = torch.randn(n, 2); path = [x.clone()]
    with torch.no_grad():
        for k in range(n_steps):
            t = torch.full((n, 1), k / n_steps); x = x + v(torch.cat([x, t], 1)) / n_steps; path.append(x.clone())
    return x, path
def nn_dist(s): return torch.cdist(s, X).min(1).values.mean().item()
out["flow"] = {}
for k in [1, 2, 4, 20]:
    s, path = sample(k); out["flow"][k] = {"pts": s.tolist(), "nn_dist": nn_dist(s)}
_, path = sample(20, n=60); out["flow_paths"] = [p.tolist() for p in path]
out["data"] = X[:800].tolist(); out["data_nn_ref"] = nn_dist(X[torch.randperm(len(X))[:800]] + 0.0)
print("flow matching mean distance to data:", {k: round(v["nn_dist"], 3) for k, v in out["flow"].items()})
json.dump(out, open("toy2d.json", "w"))
