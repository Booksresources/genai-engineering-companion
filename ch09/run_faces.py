import time, json, torch
from diff9 import *
torch.manual_seed(0); F_, L = faces(3000); betas, abar = schedule("cosine", 200); log = []; t0 = time.time()
m = UNet(16, 3); train(m, F_, L, abar, steps=2000, bs=64, log=log)
torch.save(m.state_dict(), "unet_faces.pt"); json.dump({"log": log, "seconds": time.time() - t0}, open("faces_train.json", "w"))
print("faces done", log[-1], f"{time.time()-t0:.0f}s")
