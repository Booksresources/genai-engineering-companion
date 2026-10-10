"""Text watermarking (green-list), memorization checks, and signed provenance manifests."""
import sys, json, hashlib, math, re, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/claude")
from mini_gpt import GPT, GPTConfig
res = {}
text = open("/home/claude/tiny.txt").read(); chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
m = GPT(GPTConfig(vocab_size=V)); m.load_state_dict(torch.load("/home/claude/ckpt.pt", weights_only=False)["model"]); m.eval()
def green(prev, key=42, gamma=0.5):
    g = torch.Generator().manual_seed(key * 1000 + prev); return torch.randperm(V, generator=g)[: int(gamma * V)]
@torch.no_grad()
def generate(n=300, delta=0.0, seed=0, prompt="ROMEO:\n"):
    torch.manual_seed(seed); ids = [stoi[c] for c in prompt]
    for _ in range(n):
        logits = m(torch.tensor([ids[-128:]]))[0][0, -1] / 0.8
        if delta: logits[green(ids[-1])] += delta                         # nudge toward this step's green list
        ids.append(torch.multinomial(F.softmax(logits, -1), 1).item())
    return "".join(chars[i] for i in ids[len(prompt):])
def z_score(s, gamma=0.5):
    ids = [stoi[c] for c in s]; hits = sum(ids[i] in set(green(ids[i - 1]).tolist()) for i in range(1, len(ids))); n = len(ids) - 1
    return (hits - gamma * n) / math.sqrt(n * gamma * (1 - gamma))
res["watermark"] = {}
for delta in [0.0, 1.0, 2.0, 4.0]:
    zs = [z_score(generate(delta=delta, seed=s)) for s in range(6)]
    res["watermark"][delta] = {"z_mean": float(np.mean(zs)), "z_min": float(np.min(zs))}
s_wm = generate(delta=2.0, seed=11); s_plain = generate(delta=0.0, seed=11)
def edit(s, frac, seed=0):
    rng = np.random.default_rng(seed); s = list(s)
    for i in rng.choice(len(s), int(frac * len(s)), replace=False): s[i] = chars[rng.integers(0, V)]
    return "".join(s)
res["watermark_edits"] = {f: z_score(edit(s_wm, f)) for f in [0.0, 0.1, 0.25, 0.5]}
res["examples"] = {"plain": s_plain[:120], "watermarked": s_wm[:120]}
print("watermark:", res["watermark"], "\nafter edits:", res["watermark_edits"])
# perplexity cost of watermarking
@torch.no_grad()
def nll(s):
    ids = torch.tensor([[stoi[c] for c in "ROMEO:\n" + s]])[:, :129]; return m(ids[:, :-1], ids[:, 1:])[1].item()
res["quality"] = {"nll_plain": float(np.mean([nll(generate(seed=s)) for s in range(4)])), "nll_wm_delta2": float(np.mean([nll(generate(delta=2.0, seed=s)) for s in range(4)]))}
print("quality:", res["quality"])
# ---- memorization: verbatim 30-character overlap between generated text and the training corpus
def verbatim(s, L=30): return sum(s[i:i + L] in text for i in range(0, len(s) - L, 5)) / len(range(0, len(s) - L, 5))
samples = [generate(seed=100 + s) for s in range(5)]
res["memorization"] = {"share_of_30char_windows_found_in_training": float(np.mean([verbatim(s) for s in samples])),
                       "control_real_training_text": verbatim(text[50000:50300]), "control_shuffled_words": verbatim(" ".join(np.random.default_rng(0).permutation(text[50000:50300].split())))}
print("memorization:", res["memorization"])
# ---- provenance: a signed manifest bound to the asset's hash (C2PA-style idea)
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
key = Ed25519PrivateKey.generate(); pub = key.public_key()
asset = np.random.default_rng(1).integers(0, 255, (64, 64), dtype=np.uint8).tobytes()
manifest = {"claim_generator": "Muse/1.0", "assertions": {"ai_generated": True, "model": "org/sdxl-base@rev", "reviewed_by": "j.ortiz"},
            "asset_sha256": hashlib.sha256(asset).hexdigest()}
payload = json.dumps(manifest, sort_keys=True).encode(); sig = key.sign(payload)
def verify(asset_bytes, manifest, sig):
    try: pub.verify(sig, json.dumps(manifest, sort_keys=True).encode())
    except Exception: return "invalid signature (manifest altered)"
    return "valid" if hashlib.sha256(asset_bytes).hexdigest() == manifest["asset_sha256"] else "asset altered after signing"
tampered = bytearray(asset); tampered[100] ^= 1
res["provenance"] = {"original": verify(asset, manifest, sig), "one pixel changed": verify(bytes(tampered), manifest, sig),
                     "label removed from manifest": verify(asset, {**manifest, "assertions": {**manifest["assertions"], "ai_generated": False}}, sig)}
print("provenance:", res["provenance"]); json.dump(res, open("safety2.json", "w"), indent=1)
