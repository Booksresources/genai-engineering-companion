"""Build a raw brand-copy dataset with realistic problems, then clean it."""
import json, random, re
rng = random.Random(0)
FIRST = ["Trail", "Ridge", "Ember", "Summit", "Canyon", "Harbor", "Pine", "Juniper", "Cedar", "River", "Aspen", "Granite", "Willow", "Falcon"]
SECOND = ["Mug", "Tent", "Pack", "Gloves", "Lamp", "Stove", "Bottle", "Jacket", "Boots", "Chair", "Quilt", "Poles"]
FEATS = ["packs down small", "keeps drinks hot", "fully waterproof", "fleece lined", "folds flat", "sets up fast", "weighs almost nothing",
         "built to last", "dries quickly", "fits any pack", "glows all night", "stays warm when wet", "made from recycled wool", "rated for snow"]
CLOSERS = ["Built for the trail.", "Made for long days outside.", "Ready when you are.", "Your next camp starts here."]
def copy_for(name, feat, closer): return f"Meet the {name}. {feat[0].upper() + feat[1:]}. {closer} Northwind."
def prompt_for(name, feat): return f"P: {name}; f: {feat}\nC: "
# held-out products: test names use first words never paired with these second words in training
pairs = [(f, s) for f in FIRST for s in SECOND]; rng.shuffle(pairs)
test_pairs, train_pairs = pairs[:30], pairs[30:]
raw = []
for f, s in train_pairs:
    for _ in range(7):
        feat = rng.choice(FEATS); name = f"{f} {s}"
        raw.append({"product": name, "feature": feat, "copy": copy_for(name, feat, rng.choice(CLOSERS))})
# inject realistic problems
for r in rng.sample(raw, 150): raw.append(dict(r))                                                   # exact duplicates
for r in rng.sample(raw, 60): raw.append({**r, "copy": r["copy"].replace(".", "!", 1)})               # off-brand: exclamation
for r in rng.sample(raw, 30): raw.append({**r, "copy": ""})                                          # missing output
for r in rng.sample(raw, 30): raw.append({**r, "copy": r["copy"] + " Contact sales at ann.lee@northwind.example for bulk pricing."})  # PII
for r in rng.sample(raw, 25): raw.append({**r, "copy": r["copy"] + " " + " ".join(rng.choice(CLOSERS) for _ in range(4))})   # too long
rng.shuffle(raw)
VOCAB = set("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
EMAIL = re.compile(r"[\w.]+@[\w.]+")
def clean(rows):
    report = {"raw": len(rows)}; seen, out = set(), []
    for r in rows:
        if not r["copy"].strip(): report["missing output"] = report.get("missing output", 0) + 1; continue
        if EMAIL.search(r["copy"]): report["contains personal data"] = report.get("contains personal data", 0) + 1; continue
        if "!" in r["copy"]: report["off-brand (exclamation)"] = report.get("off-brand (exclamation)", 0) + 1; continue
        if len(prompt_for(r["product"], r["feature"]) + r["copy"]) > 127: report["too long for context"] = report.get("too long for context", 0) + 1; continue
        if set(r["copy"]) - VOCAB: report["unsupported characters"] = report.get("unsupported characters", 0) + 1; continue
        key = (r["product"], r["feature"], r["copy"])
        if key in seen: report["duplicate"] = report.get("duplicate", 0) + 1; continue
        seen.add(key); out.append(r)
    report["kept"] = len(out); return out, report
train, report = clean(raw)
test = [{"product": f"{f} {s}", "feature": rng.choice(FEATS)} for f, s in test_pairs for _ in range(2)]
print(report); print(train[0]); print(len(test), "test prompts")
json.dump({"train": train, "test": test, "report": report, "closers": CLOSERS}, open("brand_data.json", "w"), indent=0)
