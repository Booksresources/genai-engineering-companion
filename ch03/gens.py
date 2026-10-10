"""Four generators on the same dataset: VAE, GAN, autoregressive, diffusion."""
import sys, time, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/claude")
from common3 import *
from mini_gpt import GPT, GPTConfig
which = sys.argv[1]; out = {}

def mlp(i, h, o, act=nn.SiLU): return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, h), act(), nn.Linear(h, o))

if which == "vae":
    class VAE(nn.Module):
        def __init__(s, z=8):
            super().__init__(); s.enc = mlp(64, 256, 2 * z); s.dec = mlp(z, 256, 64); s.z = z
        def forward(s, x):
            mu, logvar = s.enc(x).chunk(2, -1)
            zz = mu + torch.randn_like(mu) * (0.5 * logvar).exp()            # reparameterization trick
            logits = s.dec(zz)
            rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)
            kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1)
            return (rec + kl).mean(), rec.mean(), kl.mean()
        def sample(s, n): return torch.sigmoid(s.dec(torch.randn(n, s.z)))
    for z in [2, 8]:
        torch.manual_seed(0); m = VAE(z); opt = torch.optim.Adam(m.parameters(), 1e-3); it = batches(Xtr)
        hist = []
        for step in range(4000):
            loss, rec, kl = m(next(it)); opt.zero_grad(); loss.backward(); opt.step()
            if step % 200 == 0: hist.append((step, loss.item(), rec.item(), kl.item()))
        with torch.no_grad():
            test_loss = m(Xte)[0].item()
            if z == 2:
                mu = m.enc(Xte).chunk(2, -1)[0]; out["latent"] = {"mu": mu.tolist(), "y": Yte.tolist()}
                grid = torch.stack(torch.meshgrid(torch.linspace(-2.5, 2.5, 12), torch.linspace(2.5, -2.5, 12), indexing="xy"), -1).reshape(-1, 2)
                out["latent_grid"] = torch.sigmoid(m.dec(grid)).tolist()
            else:
                t0 = time.time(); S = m.sample(1000); out["sample_time_1000"] = time.time() - t0
                out["samples"] = S.tolist(); out["test_neg_elbo_nats"] = test_loss; out["hist"] = hist
        torch.save(m.state_dict(), f"vae_z{z}.pt")
    print("VAE test -ELBO (nats/image):", round(out["test_neg_elbo_nats"], 2))

if which == "gan":
    torch.manual_seed(0)
    G = nn.Sequential(nn.Linear(16, 256), nn.LeakyReLU(0.2), nn.Linear(256, 256), nn.LeakyReLU(0.2), nn.Linear(256, 64), nn.Sigmoid())
    Dn = nn.Sequential(nn.Linear(64, 256), nn.LeakyReLU(0.2), nn.Linear(256, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))
    oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999)); oD = torch.optim.Adam(Dn.parameters(), 2e-4, betas=(0.5, 0.999))
    it = batches(Xtr); hist = []
    for step in range(6000):
        real = next(it); fake = G(torch.randn(len(real), 16))
        lD = F.binary_cross_entropy_with_logits(Dn(real), torch.ones(len(real), 1)) + F.binary_cross_entropy_with_logits(Dn(fake.detach()), torch.zeros(len(real), 1))
        oD.zero_grad(); lD.backward(); oD.step()
        lG = F.binary_cross_entropy_with_logits(Dn(fake), torch.ones(len(real), 1))      # non-saturating generator loss
        oG.zero_grad(); lG.backward(); oG.step()
        if step % 200 == 0: hist.append((step, lD.item(), lG.item()))
    with torch.no_grad():
        t0 = time.time(); S = G(torch.randn(1000, 16)); out["sample_time_1000"] = time.time() - t0
    out["samples"] = S.tolist(); out["hist"] = hist; print("GAN done; final D loss %.3f G loss %.3f" % hist[-1][1:])

if which == "ar":
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=18, d_model=80, n_layers=3, n_heads=4, n_kv_heads=2, d_ff=256, max_seq_len=65))   # 17 levels + BOS
    BOS = 17
    def seq(xi): return torch.cat([torch.full((len(xi), 1), BOS), xi], 1)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=0.01); it = batches(Xtr_int, 64); hist = []
    for step in range(1000):
        for g in opt.param_groups: g["lr"] = 2e-3 * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / 1000))
        s = seq(next(it)); _, loss = m(s[:, :-1].contiguous(), s[:, 1:].contiguous())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0: hist.append((step, loss.item()))
    m.eval()
    with torch.no_grad():
        s = seq(Xte_int); _, test_loss = m(s[:, :-1].contiguous(), s[:, 1:].contiguous())
        out["test_nll_bits_per_pixel"] = test_loss.item() / math.log(2)
        t0 = time.time(); x = torch.full((1000, 1), BOS)
        for _ in range(64):                                      # one pixel at a time
            logits, _ = m(x); p = F.softmax(logits[:, -1, :17], -1); x = torch.cat([x, torch.multinomial(p, 1)], 1)
        out["sample_time_1000"] = time.time() - t0
    out["samples"] = (x[:, 1:].float() / 16).tolist(); out["hist"] = hist
    print("AR test NLL: %.3f bits/pixel" % out["test_nll_bits_per_pixel"])

if which == "diffusion":
    torch.manual_seed(0); T = 200
    betas = torch.linspace(1e-4, 0.02, T); abar = torch.cumprod(1 - betas, 0)
    class Eps(nn.Module):                                        # predicts the noise added at step t
        def __init__(s): super().__init__(); s.net = mlp(64 + 32, 512, 64)
        def forward(s, x, t):
            f = torch.exp(-math.log(1000) * torch.arange(16) / 16)
            emb = torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], 1)
            return s.net(torch.cat([x, emb], 1))
    m = Eps(); opt = torch.optim.Adam(m.parameters(), 1e-3); it = batches(Xtr * 2 - 1, 128); hist = []    # data scaled to [-1, 1]
    for step in range(8000):
        x0 = next(it); t = torch.randint(0, T, (len(x0),)); e = torch.randn_like(x0)
        xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * e        # forward process in one jump
        loss = F.mse_loss(m(xt, t.float()), e); opt.zero_grad(); loss.backward(); opt.step()
        if step % 400 == 0: hist.append((step, loss.item()))
    with torch.no_grad():
        t0 = time.time(); x = torch.randn(1000, 64)
        for t in reversed(range(T)):                              # reverse process: denoise step by step
            tt = torch.full((1000,), float(t)); eps = m(x, tt)
            x = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / (1 - betas[t]).sqrt()
            if t > 0: x += betas[t].sqrt() * torch.randn_like(x)
        out["sample_time_1000"] = time.time() - t0
    out["samples"] = ((x.clamp(-1, 1) + 1) / 2).tolist(); out["hist"] = hist; print("diffusion done; final loss %.3f" % hist[-1][1])

json.dump(out, open(f"gen_{which}.json", "w"))
