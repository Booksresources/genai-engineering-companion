import json, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from t2i import *
betas, abar = schedule("cosine", 200); rng = np.random.default_rng(9)
Xs, labs = [], []
for _ in range(2000):
    e, g, b, h = rng.integers(0, 3), rng.random() < .5, rng.random() < .5, rng.random() < .5
    Xs.append(render(e, g, b, hat=h, rng=rng).ravel()); labs.append(h)
hat_judge = LogisticRegression(max_iter=3000).fit(np.array(Xs), np.array(labs))
def hat_rate(S): return float(hat_judge.predict(S.view(len(S), -1).numpy()).mean())
# six reference images of the new concept: smiling faces, no glasses, white background, wearing a hat
r2 = np.random.default_rng(42)
refs = torch.tensor(np.array([render(2, False, False, hat=True, rng=r2) for _ in range(6)]), dtype=torch.float32)[:, None]
ref_ids = tokenize([caption(2, False, False, concept="sks")] * 6)
Xp, Cp, _ = dataset(1000, seed=5); prior_ids = tokenize(Cp)
test_new = [caption(e, g, b, concept="sks") for e, g, b in [(0, True, True), (1, False, True), (0, False, False), (1, True, False)] * 15]
test_old = [caption(e, g, b) for e, g, b in [(0, True, True), (1, False, True), (2, False, False), (1, True, False)] * 15]
res = {"refs": 6}
for prior in [False, True, "balanced"]:
    torch.manual_seed(0); m = TextUNet(); m.load_state_dict(torch.load("t2i.pt"))
    reps = 170 if prior == "balanced" else 50
    X = torch.cat([refs] * reps + ([Xp] if prior else [])); I = torch.cat([ref_ids] * reps + ([prior_ids] if prior else []))
    train(m, X, I, abar, steps=400 if prior == "balanced" else 300, lr=5e-4, p_uncond=0.1, seed=1)
    m.eval(); Sn = generate(m, test_new, abar, guidance=4.0); So = generate(m, test_old, abar, guidance=4.0)
    key = {False: "without prior preservation", True: "with prior preservation (1:4)", "balanced": "with prior preservation (1:1)"}[prior]
    res[key] = {"hat_rate_new_prompts": hat_rate(Sn), "hat_leak_old_prompts": hat_rate(So)}
    torch.save({"new": Sn[:4], "old": So[:4], "prompts_new": test_new[:4], "prompts_old": test_old[:4]}, f"db_{prior}.pt")
    print(key, res[key])
m = TextUNet(); m.load_state_dict(torch.load("t2i.pt")); m.eval()
res["before fine-tuning"] = {"hat_rate_new_prompts": hat_rate(generate(m, test_new, abar, guidance=4.0)), "hat_leak_old_prompts": hat_rate(generate(m, test_old, abar, guidance=4.0))}
print("before", res["before fine-tuning"]); torch.save(refs, "db_refs.pt")
json.dump(res, open("dreambooth.json", "w"), indent=1)
