import sys, time, json, torch
from diff9 import *
torch.manual_seed(0); X, Y = digits(); betas, abar = schedule("cosine", 200)
which = sys.argv[1]; log = []; t0 = time.time()
m = UNet(16, 10) if which == "unet" else DiT(d=64, depth=3)
train(m, X[:1500], Y[:1500], abar, steps=2000, bs=64, log=log)
torch.save(m.state_dict(), f"{which}_digits.pt")
json.dump({"log": log, "params": sum(p.numel() for p in m.parameters()), "seconds": time.time() - t0}, open(f"{which}_train.json", "w"))
print(which, "params", sum(p.numel() for p in m.parameters()), "final loss", log[-1], f"{time.time()-t0:.0f}s")
