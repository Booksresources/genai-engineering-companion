import time, json, torch
from t2i import *
torch.manual_seed(0); X, C, A = dataset(4000); ids = tokenize(C); betas, abar = schedule("cosine", 200)
m = TextUNet(); log = []; t0 = time.time(); train(m, X, ids, abar, steps=2200, log=log)
torch.save(m.state_dict(), "t2i.pt"); json.dump({"log": log, "seconds": time.time() - t0, "params": sum(p.numel() for p in m.parameters()), "example_captions": C[:4]}, open("t2i_train.json", "w"))
print("params", sum(p.numel() for p in m.parameters()), "loss", log[-1], f"{time.time()-t0:.0f}s", C[:2])
