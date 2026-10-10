import json, numpy as np, torch
from scipy import linalg
from sklearn.neural_network import MLPClassifier
from common3 import Xtr, Xte, Ytr, Yte
clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=0).fit(Xtr.numpy(), Ytr.numpy())
print("classifier test accuracy on real digits: %.3f" % clf.score(Xte.numpy(), Yte.numpy()))
def feats(X): return np.maximum(0, X @ clf.coefs_[0] + clf.intercepts_[0])             # hidden-layer features
def frechet(A, B):
    mu1, mu2 = A.mean(0), B.mean(0); s1, s2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    cs = linalg.sqrtm(s1 @ s2).real; return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))
real_te = Xte.numpy(); tr = Xtr.numpy()
def nn_train(S): return np.sqrt(((S[:, None, :] - tr[None]) ** 2).sum(-1)).min(1).mean()
ref_nn = nn_train(real_te)
def evaluate(S):
    P = clf.predict_proba(S); cls = P.argmax(1); counts = np.bincount(cls, minlength=10) / len(cls)
    ent = -(counts[counts > 0] * np.log2(counts[counts > 0])).sum()
    return {"confidence": float(P.max(1).mean()), "class_entropy_bits": float(ent), "fd": frechet(feats(S), feats(real_te)),
            "nn_ratio": float(nn_train(S) / ref_nn), "class_counts": counts.tolist()}
res = {"real (held-out test)": evaluate(real_te)}
res["real (held-out test)"]["fd"] = frechet(feats(tr[:297]), feats(real_te))      # baseline: real vs real
for name in ["vae", "gan", "ar", "diffusion"]:
    g = json.load(open(f"gen_{name}.json")); S = np.array(g["samples"])
    r = evaluate(S); r["sample_time_1000"] = g["sample_time_1000"]
    r["likelihood"] = g.get("test_nll_bits_per_pixel", g.get("test_neg_elbo_nats")); res[name] = r
for k, v in res.items(): print(f"{k:22s} conf {v['confidence']:.3f}  classH {v['class_entropy_bits']:.2f}  FD {v['fd']:.1f}  NNratio {v['nn_ratio']:.2f}  time {v.get('sample_time_1000', 0):.2f}s")
json.dump(res, open("metrics.json", "w"))
