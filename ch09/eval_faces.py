import json, time, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from diff9 import *
betas, abar = schedule("cosine", 200); F_, L = faces(3000); Ft, Lt = faces(600, seed=1)
clf = LogisticRegression(max_iter=2000).fit(F_.view(3000, -1).numpy(), L.numpy())
print("expression classifier accuracy on new real faces:", clf.score(Ft.view(600, -1).numpy(), Lt.numpy()))
m = UNet(16, 3); m.load_state_dict(torch.load("unet_faces.pt")); m.eval()
res = {}
y = torch.arange(300) % 3
for w in [1.0, 3.0]:
    S = sample(m, 300, y, betas, abar, steps=20, guidance=w, shape=(1, 16, 16))
    acc = (clf.predict(S.view(300, -1).numpy()) == y.numpy()).mean()
    nn_d = torch.cdist(S.view(300, -1), F_.view(3000, -1)).min(1).values.mean().item()
    ref_d = torch.cdist(Ft.view(600, -1)[:300], F_.view(3000, -1)).min(1).values.mean().item()
    res[w] = {"expression_match": float(acc), "nn_ratio": nn_d / ref_d}; torch.save(S[:24], f"faces_w{w}.pt")
    print("guidance", w, res[w])
# denoising trajectory for 3 faces (one per expression)
g = torch.Generator().manual_seed(5); x = torch.randn((3, 1, 16, 16), generator=g); yy = torch.tensor([0, 1, 2]); ts = torch.linspace(199, 0, 20).long().tolist(); traj = [x.clone()]
with torch.no_grad():
    for i, t in enumerate(ts):
        tt = torch.full((3,), t); e_c, e_u = m(x, tt, yy), m(x, tt, torch.full_like(yy, 3)); e = e_u + 3.0 * (e_c - e_u)
        a_t = abar[t]; a_p = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x0 = ((x - (1 - a_t).sqrt() * e) / a_t.sqrt()).clamp(-1, 1); x = a_p.sqrt() * x0 + (1 - a_p).sqrt() * e; traj.append(x.clone())
torch.save(torch.stack([traj[k] for k in [0, 4, 8, 12, 16, 20]]), "trajectory.pt")
# a small autoencoder: 16x16 pixels -> 4x4x4 latent (4x fewer numbers), the idea behind latent diffusion
torch.manual_seed(0)
enc = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 4, 4, 2, 1))
dec = nn.Sequential(nn.ConvTranspose2d(4, 32, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh())
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 2e-3); t0 = time.time()
for s in range(1500):
    xb = F_[torch.randint(0, 3000, (64,))]; loss = F.mse_loss(dec(enc(xb)), xb); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    z = enc(Ft[:8]); rec = dec(z); mse = F.mse_loss(dec(enc(Ft)), Ft).item()
res["autoencoder"] = {"latent_shape": list(z.shape[1:]), "pixels": 256, "latent_numbers": int(z[0].numel()), "test_mse": mse, "seconds": time.time() - t0}
torch.save({"orig": Ft[:8], "rec": rec}, "ae_recon.pt"); print(res["autoencoder"])
json.dump(res, open("faces_eval.json", "w"), indent=1)
