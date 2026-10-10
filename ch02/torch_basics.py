import torch, torch.nn as nn, json
torch.manual_seed(0)
# 1) Tensors, dtypes, devices
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(4, 3)                                  # float32 by default
print("shape", tuple(x.shape), "dtype", x.dtype, "device", device)
print("bf16 bytes per element:", x.to(torch.bfloat16).element_size())
# 2) Autograd
w = torch.tensor(2.0, requires_grad=True)
loss = (w * 3.0 - 9.0) ** 2; loss.backward()
print("d loss / d w at w=2:", w.grad.item())            # 2*(3w-9)*3 = -18
# 3) A module, a loss, an optimizer, and a training loop: learn a noisy 2-D spiral classifier
n = 400; t = torch.linspace(0, 3 * torch.pi, n)
X = torch.cat([torch.stack([t * torch.cos(t), t * torch.sin(t)], 1), torch.stack([-t * torch.cos(t), -t * torch.sin(t)], 1)]) / 10
X += 0.03 * torch.randn_like(X); y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
model = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
hist = []
for step in range(400):
    logits = model(X.to(device)); loss = nn.functional.cross_entropy(logits, y.to(device))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 20 == 0 or step == 399:
        acc = (logits.argmax(1).cpu() == y).float().mean().item(); hist.append((step, loss.item(), acc))
print("final loss %.3f accuracy %.3f" % hist[-1][1:])
# 4) Save and reload
torch.save(model.state_dict(), "spiral.pt")
m2 = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2)); m2.load_state_dict(torch.load("spiral.pt"))
print("reloaded model agrees:", torch.equal(m2(X).argmax(1), model.cpu()(X).argmax(1)))
print("parameters:", sum(p.numel() for p in model.parameters()))
json.dump({"hist": hist, "X": X.tolist(), "y": y.tolist()}, open("torch_basics.json", "w"))
# grid predictions for a figure
g = torch.stack(torch.meshgrid(torch.linspace(-1.1, 1.1, 120), torch.linspace(-1.1, 1.1, 120), indexing="xy"), -1).reshape(-1, 2)
json.dump({"grid": g.tolist(), "pred": torch.softmax(m2(g), 1)[:, 1].tolist()}, open("grid.json", "w"))
