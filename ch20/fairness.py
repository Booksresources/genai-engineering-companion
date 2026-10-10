"""Counterfactual fairness test of the brand-safety classifier: same post, different audience words."""
import json, numpy as np, re
exec(open("safety1.py").read().split("# ---- 1)")[0])                       # imports only
import importlib.util, sys
src = open("safety1.py").read(); src = src[src.index("# ---- 2)"):src.index("missed =")]
ns = {}; exec("import random, numpy as np\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\nres={}\n" + src, ns)
clf, vec = ns["clf"], ns["vec"]
base = ["Take the {p} on your next morning hike, {a}.", "The {p} keeps drinks hot for six hours, {a}.", "Warm hands and early starts with the {p}, {a}."]
groups = {"no audience": "for everyone", "women": "made for women", "men": "made for men", "older hikers": "made for older hikers", "kids": "made for kids",
          "wheelchair users": "made for wheelchair users"}
P = ["TrailMug", "Ember Gloves", "Ridge Tent", "Summit Pack"]
out = {}
for g, a in groups.items():
    xs = [b.format(p=p, a=a) for b in base for p in P]; pr = clf.predict_proba(vec.transform(xs))[:, 1]
    out[g] = {"mean_unsafe_score": float(pr.mean()), "blocked_at_0.5": float((pr >= 0.5).mean())}
print(json.dumps(out, indent=1)); json.dump(out, open("fairness.json", "w"), indent=1)
