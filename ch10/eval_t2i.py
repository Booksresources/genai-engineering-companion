import json, time, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from t2i import *
betas, abar = schedule("cosine", 200)
rng = np.random.default_rng(9)
# attribute judges trained on real renders (including hats)
Xs, labs = [], []
for _ in range(3000):
    e, g, b, h = rng.integers(0, 3), rng.random() < .5, rng.random() < .5, rng.random() < .5
    Xs.append(render(e, g, b, hat=h, rng=rng).ravel()); labs.append((e, g, b, h))
Xs = np.array(Xs); labs = np.array(labs)
judges = [LogisticRegression(max_iter=3000).fit(Xs[:2500], labs[:2500, k]) for k in range(4)]
print("judge accuracy on held-out renders:", [round(j.score(Xs[2500:], labs[2500:, k]), 3) for k, j in enumerate(judges)])
def judge(S): v = S.view(len(S), -1).numpy(); return np.stack([j.predict(v) for j in judges], 1)
m = TextUNet(); m.load_state_dict(torch.load("t2i.pt")); m.eval()
rng2 = np.random.default_rng(3); req = [(rng2.integers(0, 3), rng2.random() < .5, rng2.random() < .5) for _ in range(240)]
prompts = [caption(*r) for r in req]; R = np.array([(e, g, b) for e, g, b in req])
res = {"guidance": {}}
for w in [1.0, 2.0, 4.0, 7.0]:
    S = generate(m, prompts, abar, guidance=w); J = judge(S)[:, :3]
    match = (J == R).mean(0); res["guidance"][w] = {"expression": float(match[0]), "glasses": float(match[1]), "background": float(match[2]), "all": float((J == R).all(1).mean())}
    if w == 4.0: torch.save({"S": S[:12], "prompts": prompts[:12]}, "t2i_samples.pt")
    print("w", w, res["guidance"][w])
# negative prompt: ask for a face, steer away from glasses
p = ["a smiling face with glasses on white background"] * 60
S1 = generate(m, ["a smiling face on white background"] * 60, abar, guidance=4.0)
S2 = generate(m, ["a smiling face on white background"] * 60, abar, guidance=4.0, negative=["a face with glasses"] * 60)
res["negative"] = {"glasses_rate_plain": float(judge(S1)[:, 1].mean()), "glasses_rate_with_negative": float(judge(S2)[:, 1].mean())}
print("negative prompt:", res["negative"])
# inpainting: regenerate the lower half of real frowning faces as smiling
known = torch.tensor(np.array([render(0, False, False, rng=np.random.default_rng(100 + i)) for i in range(60)]), dtype=torch.float32)[:, None]
mask = torch.zeros(1, 1, 16, 16); mask[:, :, 9:, :] = 1
S3 = generate(m, ["a smiling face without glasses on white background"] * 60, abar, guidance=4.0, mask=mask, known=known)
res["inpaint"] = {"smiling_after": float((judge(S3)[:, 0] == 2).mean()), "top_half_unchanged": bool(torch.allclose(S3[:, :, :9], known[:, :, :9]))}
torch.save({"known": known[:6], "out": S3[:6]}, "inpaint.pt"); print("inpaint:", res["inpaint"])
# image-to-image: change background with different strengths
init = torch.tensor(np.array([render(2, True, False, rng=np.random.default_rng(200 + i)) for i in range(60)]), dtype=torch.float32)[:, None]
res["img2img"] = {}; keep = {}
for sres in [0.3, 0.5, 0.7, 1.0]:
    S4 = generate(m, ["a smiling face with glasses on grey background"] * 60, abar, guidance=4.0, init=init, strength=sres, steps=max(4, int(20 * sres)))
    J = judge(S4); res["img2img"][sres] = {"grey_background": float(J[:, 2].mean()), "pixel_change": float((S4 - init).abs().mean())}
    keep[sres] = S4[:4]
torch.save({"init": init[:4], **{str(k): v for k, v in keep.items()}}, "img2img.pt"); print("img2img:", res["img2img"])
json.dump(res, open("t2i_eval.json", "w"), indent=1)
