import sys, json, re, math, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/claude")
from audio import SR, demo_signal, snr_db
res = {}
# ---- 1) word error rate with normalization
def normalize(s): return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
def wer(ref, hyp):
    r, h = normalize(ref), normalize(hyp); d = np.zeros((len(r) + 1, len(h) + 1), int); d[:, 0] = range(len(r) + 1); d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(len(r), 1)
ref = "The TrailMug keeps drinks hot for six hours."
hyps = {"exact": "the trailmug keeps drinks hot for six hours", "number as digits": "The TrailMug keeps drinks hot for 6 hours.",
        "brand misheard": "The trail mug keeps drinks hot for six hours.", "dropped word": "The TrailMug keeps drinks hot six hours."}
res["wer"] = {k: wer(ref, v) for k, v in hyps.items()}; print("WER:", res["wer"])
# ---- 2) spread-spectrum audio watermark
def embed(x, key, strength=0.004, chip=64):
    rng = np.random.default_rng(key); pn = rng.choice([-1.0, 1.0], len(x)); return x + strength * pn * np.abs(x).mean() / 0.15, pn
def detect(y, key):
    pn = np.random.default_rng(key).choice([-1.0, 1.0], len(y)); return float((y * pn).sum() / (np.sqrt((y ** 2).sum()) * np.sqrt(len(y))) * np.sqrt(len(y)))
clean = demo_signal(seconds=3.0)
marked, _ = embed(clean, key=1234)
res["watermark"] = {"embedding_snr_db": snr_db(clean, marked)}
tests = {"marked": marked, "unmarked": clean, "marked + noise": marked + np.random.default_rng(1).normal(0, 0.02, len(marked)),
         "marked, volume x0.5": marked * 0.5, "marked, 8 kHz resample": np.interp(np.arange(len(marked)), np.arange(0, len(marked), 2), marked[::2]),
         "marked, wrong key": None}
scores = {}
for k, y in tests.items():
    scores[k] = detect(marked, key=999) if y is None else detect(y, key=1234)
null = [detect(demo_signal(3.0, seed=s) + np.random.default_rng(s).normal(0, 0.01, len(clean)), key=1234) for s in range(200)]
res["watermark"].update({"scores": scores, "null_mean": float(np.mean(null)), "null_std": float(np.std(null)), "threshold": 4.0})
print("watermark:", res["watermark"])
# ---- 3) a tiny autoregressive music model over note tokens
from mini_gpt import GPT, GPTConfig
SCALE = [60, 62, 64, 65, 67, 69, 71, 72]                       # C major, MIDI numbers
rng = np.random.default_rng(0)
def melody(n=32):
    notes, i = [], rng.integers(0, 8)
    for _ in range(n):
        i = int(np.clip(i + rng.choice([-2, -1, -1, 0, 1, 1, 2]), 0, 7)); dur = rng.choice([1, 1, 2])
        notes.append(SCALE[i] - 48); notes.append(24 + dur)     # pitch token (12..24), duration token (25..26)
    return notes
data = torch.tensor([melody() for _ in range(1500)])
m = GPT(GPTConfig(vocab_size=28, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=176, max_seq_len=64))
opt = torch.optim.AdamW(m.parameters(), 3e-3); t0 = time.time()
for s in range(600):
    b = data[torch.randint(0, 1500, (32,))]; _, loss = m(b[:, :-1].contiguous(), b[:, 1:].contiguous())
    opt.zero_grad(); loss.backward(); opt.step()
torch.manual_seed(1); gen = m.generate(torch.tensor([[SCALE[0] - 48]]), 63, temperature=1.0)[0].tolist()
pitches = [p + 48 for p in gen[0::2] if p < 25]
rand = list(np.random.default_rng(2).integers(48, 84, len(pitches)))
in_key = lambda ps: float(np.mean([p in SCALE for p in ps]))
step = lambda ps: float(np.mean(np.abs(np.diff(ps))))
res["music"] = {"train_seconds": time.time() - t0, "final_loss": loss.item(), "generated_in_key": in_key(pitches), "random_in_key": in_key(rand),
                "generated_mean_step": step(pitches), "random_mean_step": step(rand), "pitches": pitches[:24]}
wave = np.concatenate([0.2 * np.sin(2 * np.pi * 440 * 2 ** ((p - 69) / 12) * np.arange(int(SR * 0.2)) / SR) * np.hanning(int(SR * 0.2)) for p in pitches[:16]])
import wave as wv
with wv.open("melody.wav", "w") as f: f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes((wave * 32767).astype(np.int16).tobytes())
print("music:", {k: v for k, v in res["music"].items() if k != "pitches"})
# ---- 4) loudness normalization for voiceovers (RMS-based approximation)
def normalize_rms(x, target_dbfs=-20.0, peak_limit=0.95):
    rms = np.sqrt((x ** 2).mean()); g = 10 ** (target_dbfs / 20) / max(rms, 1e-9); y = x * g
    return y * min(1, peak_limit / np.abs(y).max())
clips = [demo_signal(1.5, seed=s) * gain for s, gain in enumerate([0.2, 1.0, 2.5])]
res["loudness"] = {"before_dbfs": [float(20 * np.log10(np.sqrt((c ** 2).mean()))) for c in clips],
                   "after_dbfs": [float(20 * np.log10(np.sqrt((normalize_rms(c) ** 2).mean()))) for c in clips]}
print("loudness:", res["loudness"]); json.dump(res, open("audio2.json", "w"), indent=1)
