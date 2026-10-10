import json, time, torch
from video import *
betas, abar = schedule("cosine", 200); res = {"real": analyze(clips(200, seed=7))}
for which in ["frame", "video"]:
    m = Denoiser(temporal=(which == "video")); m.load_state_dict(torch.load(f"{which}.pt")); m.eval()
    t0 = time.time(); Vs = sample(m, 200, abar); dt = time.time() - t0
    res[which] = {**analyze(Vs), "seconds_200_clips": dt}; torch.save(Vs[:4], f"samples_{which}.pt"); print(which, res[which])
m = Denoiser(True); m.load_state_dict(torch.load("video.pt")); m.eval()
real = clips(200, seed=11); Vi = sample(m, 200, abar, first_frame=real[:, :, 0])
res["image_to_video"] = analyze(Vi); torch.save({"real": real[:4], "gen": Vi[:4]}, "i2v.pt"); print("i2v", res["image_to_video"])
print("real", res["real"]); json.dump(res, open("video_eval.json", "w"), indent=1)
