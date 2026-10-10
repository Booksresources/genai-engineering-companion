import sys, json, math, time, torch
sys.path.insert(0, "/home/claude")
from mini_gpt import GPT, GPTConfig
torch.manual_seed(0)
text = open("/home/claude/tiny.txt").read(); chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text]); n = int(0.9 * len(data)); tr, va = data[:n], data[n:]
cfg = GPTConfig(vocab_size=len(chars), max_seq_len=128)
model = GPT(cfg); print("parameters:", sum(p.numel() for p in model.parameters()))
def get_batch(split, bs=32, T=128):
    d = tr if split == "train" else va; ix = torch.randint(0, len(d) - T - 1, (bs,))
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])
@torch.no_grad()
def val_loss():
    model.eval(); g = torch.Generator().manual_seed(1); tot = 0
    for _ in range(10):
        ix = torch.randint(0, len(va) - 129, (32,), generator=g)
        x = torch.stack([va[i:i + 128] for i in ix]); y = torch.stack([va[i + 1:i + 129] for i in ix]); tot += model(x, y)[1].item()
    model.train(); return tot / 10
def sample(n=90):
    torch.manual_seed(7); out = model.generate(torch.tensor([[stoi[c] for c in "ROMEO:\n"]]), n, temperature=0.8, top_k=20)
    return "".join(chars[i] for i in out[0, 7:].tolist())
STEPS = int(sys.argv[1]); opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
log, samples = [], {0: sample()}; t0 = time.time()
for step in range(1, STEPS + 1):
    lr = 3e-3 * min(1, step / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
    for g in opt.param_groups: g["lr"] = lr
    x, y = get_batch("train"); _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 100 == 0: log.append((step, loss.item(), val_loss()))
    if step in (100, 300, STEPS // 2, STEPS): samples[step] = sample()
print(f"{STEPS} steps in {time.time()-t0:.0f}s; final train {log[-1][1]:.3f} val {log[-1][2]:.3f}")
for k, v in samples.items(): print(k, repr(v[:80]))
json.dump({"log": log, "samples": samples, "seconds": time.time() - t0}, open("train_tiny.json", "w"))
