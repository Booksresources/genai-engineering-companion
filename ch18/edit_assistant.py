"""An image-editing assistant: instruction -> structured edit call (schema-validated) -> diffusion edit -> automatic check."""
import sys, json, re, numpy as np, torch
sys.path.insert(0, "/home/claude/gb/ch10"); sys.path.insert(0, "/home/claude/gb/ch9")
from t2i import TextUNet, generate, render, caption
from diff9 import schedule
from jsonschema import validate
from sklearn.linear_model import LogisticRegression
EDIT_SCHEMA = {"type": "object", "required": ["operation", "target", "value"], "additionalProperties": False,
               "properties": {"operation": {"enum": ["inpaint", "restyle"]}, "target": {"enum": ["mouth", "eyes", "background"]},
                              "value": {"type": "string"}}}
def parse_instruction(text):
    """Stand-in for an LLM function call that turns a request into a structured edit."""
    t = text.lower()
    if "smil" in t: call = {"operation": "inpaint", "target": "mouth", "value": "smiling"}
    elif "frown" in t or "sad" in t: call = {"operation": "inpaint", "target": "mouth", "value": "frowning"}
    elif "glasses" in t: call = {"operation": "inpaint", "target": "eyes", "value": "with" if "remove" not in t else "without"}
    elif "grey" in t or "gray" in t: call = {"operation": "restyle", "target": "background", "value": "grey"}
    else: raise ValueError("unsupported edit")
    validate(call, EDIT_SCHEMA); return call
MASKS = {"mouth": (slice(9, 16), slice(0, 16)), "eyes": (slice(3, 9), slice(0, 16))}
def apply_edit(model, abar, img, attrs, call, seed=0):
    e, g, b = attrs
    if call["target"] == "mouth": e = {"frowning": 0, "neutral": 1, "smiling": 2}[call["value"]]
    if call["target"] == "eyes": g = call["value"] == "with"
    if call["target"] == "background": b = True
    prompt = caption(e, g, b)
    if call["operation"] == "inpaint":
        mask = torch.zeros(1, 1, 16, 16); r, c = MASKS[call["target"]]; mask[:, :, r, c] = 1
        return generate(model, [prompt], abar, guidance=4.0, mask=mask, known=img[None], seed=seed)[0], (e, g, b)
    return generate(model, [prompt], abar, guidance=4.0, init=img[None], strength=0.5, steps=10, seed=seed)[0], (e, g, b)
if __name__ == "__main__":
    betas, abar = schedule("cosine", 200); m = TextUNet(); m.load_state_dict(torch.load("/home/claude/gb/ch10/t2i.pt")); m.eval()
    rng = np.random.default_rng(9); Xs, labs = [], []
    for _ in range(3000):
        e_, g_, b_ = rng.integers(0, 3), rng.random() < .5, rng.random() < .5; Xs.append(render(e_, g_, b_, rng=rng).ravel()); labs.append((e_, g_, b_))
    Xs, labs = np.array(Xs), np.array(labs); judges = [LogisticRegression(max_iter=3000).fit(Xs, labs[:, k]) for k in range(3)]
    def attrs_of(x): v = x.view(1, -1).numpy(); return tuple(int(j.predict(v)[0]) for j in judges)
    requests = ["Make the face smile", "Add glasses please", "Put it on a grey background"]
    res = {"parsed": {r: parse_instruction(r) for r in requests}, "results": {}}
    examples = {}
    for req in requests:
        call = parse_instruction(req); ok = kept = 0
        for i in range(40):
            r2 = np.random.default_rng(500 + i); a0 = (0, False, False)
            img = torch.tensor(render(*a0, rng=r2), dtype=torch.float32)[None]
            out, want = apply_edit(m, abar, img, a0, call, seed=i); got = attrs_of(out)
            k = {"mouth": 0, "eyes": 1, "background": 2}[call["target"]]
            ok += got[k] == int(want[k]); kept += all(got[j] == int(want[j]) for j in range(3) if j != k)
            if i < 2: examples.setdefault(req, []).append((img[0].tolist(), out[0].tolist()))
        res["results"][req] = {"edit_succeeded": ok / 40, "other_attributes_kept": kept / 40}; print(req, call, res["results"][req])
    try: parse_instruction("Make it look like a famous painting")
    except ValueError as e: res["unsupported"] = str(e); print("unsupported ->", e)
    json.dump(res, open("edit.json", "w"), indent=1); torch.save(examples, "edit_examples.pt")
