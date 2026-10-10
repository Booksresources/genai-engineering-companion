import numpy as np
from scipy import linalg
from sklearn.neural_network import MLPClassifier
from diff9 import digits
X, Y = digits(); Xn = ((X.view(len(X), -1).numpy() + 1) / 2)
clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=0).fit(Xn[:1500], Y[:1500].numpy())
def feats(S): return np.maximum(0, S @ clf.coefs_[0] + clf.intercepts_[0])
def fd(S, R=None):
    A, B_ = feats(S), feats(Xn[1500:] if R is None else R)
    mu1, mu2 = A.mean(0), B_.mean(0); s1, s2 = np.cov(A, rowvar=False), np.cov(B_, rowvar=False)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * linalg.sqrtm(s1 @ s2).real))
def score(samples, requested=None):
    S = ((samples.view(len(samples), -1).numpy() + 1) / 2); P = clf.predict_proba(S)
    out = {"fd": fd(S), "confidence": float(P.max(1).mean())}
    if requested is not None: out["class_match"] = float((P.argmax(1) == requested.numpy()).mean())
    return out
