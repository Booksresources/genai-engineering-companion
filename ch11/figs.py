import sys, json, torch
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch11")
A = json.load(open("audio.json")); A2 = json.load(open("audio2.json")); x = np.load("demo.npy"); sp = torch.load("spec.pt"); SR = 16000

# 11.1 representations
fig, axes = plt.subplots(3, 1, figsize=(12, 8.4), sharex=False)
t = np.arange(len(x)) / SR
axes[0].plot(t, x, color=L_BLUE, lw=0.6); axes[0].set_ylabel("amplitude"); axes[0].set_title("(a) Waveform: 16,000 samples per second"); axes[0].set_xlim(0, t[-1])
for ax_, (lab, tx) in zip([axes[0]] * 3, [("chord", 0.25), ("rising tone", 0.75), ("noise burst", 1.17)]): ax_.text(tx, 0.62, lab, ha="center", fontsize=13, fontweight="bold")
axes[0].set_ylim(-0.75, 0.8)
mag = sp["mag"].numpy(); axes[1].imshow(20 * np.log10(mag + 1e-6), origin="lower", aspect="auto", cmap="magma", extent=(0, t[-1], 0, SR / 2 / 1000))
axes[1].set_ylabel("kHz"); axes[1].set_title("(b) Spectrogram (STFT magnitude, 257 frequency bins)")
mel = sp["mel"].numpy(); axes[2].imshow(10 * np.log10(mel + 1e-8), origin="lower", aspect="auto", cmap="magma", extent=(0, t[-1], 0, 64))
axes[2].set_ylabel("mel band"); axes[2].set_xlabel("seconds"); axes[2].set_title("(c) Mel spectrogram (64 bands, spaced like human pitch perception)")
for a in axes: clean(a)
fig.tight_layout(); save(fig, "fig_11_1_representations")

# 11.2 codecs
ce = torch.load("codec_example.pt")
fig = plt.figure(figsize=(13, 5.6))
ax = fig.add_axes([0.0, 0.52, 0.55, 0.46]); ax.set_xlim(0, 66); ax.set_ylim(0, 24); ax.axis("off")
box(ax, 0.5, 6, 11, 12, "waveform", C_GREY, fs=13); box(ax, 14, 6, 11, 12, "encoder", C_LAV, fs=13); box(ax, 28, 6, 13, 12, "codebook → tokens", C_AMBER, fs=13)
box(ax, 44, 6, 11, 12, "decoder", C_LAV, fs=13); box(ax, 57.5, 6, 8, 12, "audio", C_GREY, fs=13)
for a_, b_ in [(11.7, 14), (25.2, 28), (41.2, 44), (55.2, 57.5)]: arrow(ax, (a_, 12), (b_, 12))
a2 = fig.add_axes([0.06, 0.1, 0.46, 0.36])
a2.plot(ce["orig"].numpy()[:600], color=L_BLUE, lw=1.4, label="original"); a2.plot(ce["rec"].numpy()[:600], color=L_RED, lw=1.2, label="decoded from tokens")
a2.set_xlabel("sample"); a2.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.3)); clean(a2)
a3 = fig.add_axes([0.63, 0.15, 0.35, 0.75])
kb = [256, A["mu_law_8bit"]["kbps"], A["vq_codec"]["kbps"]]
a3.bar(["PCM", "μ-law", "VQ codec"], kb, color=[C_GREY, C_BLUE, C_AMBER], edgecolor=EDGE); a3.set_yscale("log")
for i, v in enumerate(kb): a3.text(i, v * 1.25, f"{v:g} kbps", ha="center", fontsize=13, fontweight="bold")
a3.set_ylabel("Bitrate (kbps, log)"); a3.set_ylim(0.5, 1200); clean(a3)
from matplotlib.ticker import FuncFormatter; a3.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
save(fig, "fig_11_2_codec")

# 11.3 ASR + TTS pipelines
fig, ax = canvas(13, 6.8, 124, 62)
label(ax, 62, 59, "Speech recognition (speech → text)", fs=15, bold=True)
st = [("audio", C_GREY), ("log-mel spectrogram", C_BLUE), ("encoder (Transformer)", C_LAV), ("decoder → text tokens", C_LAV), ("normalize: numbers, names, casing", C_AMBER)]
for i, (t_, fc) in enumerate(st):
    xx = 1 + i * 24.6; box(ax, xx, 40, 21.5, 14, t_, fc, fs=13.5)
    if i < 4: arrow(ax, (xx + 21.7, 47), (xx + 24.6, 47))
label(ax, 62, 30, "Text-to-speech (text → speech)", fs=15, bold=True)
st2 = [("script + SSML hints", C_GREY), ("text normalization and phonemes", C_AMBER), ("acoustic model or audio-token LM (+ voice embedding)", C_LAV), ("vocoder / codec decoder", C_TEAL), ("audio", C_GREEN)]
for i, (t_, fc) in enumerate(st2):
    xx = 1 + i * 24.6; box(ax, xx, 8, 21.5, 16, t_, fc, fs=13)
    if i < 4: arrow(ax, (xx + 21.7, 16), (xx + 24.6, 16))
save(fig, "fig_11_3_pipelines")

# 11.4 music piano roll
mu = A2["music"]; ps = mu["pitches"]; rnd = list(np.random.default_rng(2).integers(48, 84, len(ps)))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
SCALE = [60, 62, 64, 65, 67, 69, 71, 72]
for ax_, seq, title, col in [(axes[0], ps, f"Generated: {100*mu['generated_in_key']:.0f}% in key, jump {mu['generated_mean_step']:.1f}", L_TEAL),
                             (axes[1], rnd, f"Random: {100*mu['random_in_key']:.0f}% in key, jump {mu['random_mean_step']:.1f}", L_RED)]:
    for s in SCALE: ax_.axhline(s, color="#DDD", lw=1)
    ax_.bar(range(len(seq)), [0.8] * len(seq), bottom=np.array(seq) - 0.4, width=0.85, color=col, edgecolor=EDGE)
    ax_.set_title(title, fontsize=13.5); ax_.set_xlabel("note index"); clean(ax_)
axes[0].set_ylabel("MIDI pitch")
fig.tight_layout(w_pad=2); save(fig, "fig_11_4_music")

# 11.5 watermark
wm = A2["watermark"]; sc = wm["scores"]; thr = wm["null_mean"] + 5 * wm["null_std"]
fig, ax = plt.subplots(figsize=(12, 5.4))
names = list(sc.keys()); vals = [sc[n] for n in names]
ax.barh(range(len(names))[::-1], vals, color=[C_GREEN if v > thr else C_ROSE for v in vals], edgecolor=EDGE)
for i, v in enumerate(vals): ax.text(v + 0.05, len(names) - 1 - i, f"{v:.2f}", va="center", fontsize=13, fontweight="bold")
ax.axvline(thr, color=BLACK, ls="--", lw=1.6); ax.text(thr + 0.05, len(names) - 0.4, f"threshold {thr:.2f} (5σ above unmarked audio)", fontsize=12.5)
ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names, fontsize=13.5); ax.set_xlabel("Detection score"); ax.set_xlim(0, 4.2); clean(ax)
save(fig, "fig_11_5_watermark")

# 11.6 Muse voiceover pipeline + loudness
ld = A2["loudness"]
fig = plt.figure(figsize=(13, 5.4))
ax = fig.add_axes([0, 0.05, 0.62, 0.9]); ax.set_xlim(0, 76); ax.set_ylim(0, 54); ax.axis("off")
st = [("approved script (Ch. 5)", C_GREY), ("consented brand voice", C_LAV), ("TTS API or model", C_BLUE), ("ASR check: WER vs. script", C_AMBER), ("loudness normalize", C_TEAL), ("watermark + label", C_GREEN)]
for i, (t_, fc) in enumerate(st):
    col = i if i < 3 else 5 - i
    xx = 1 + col * 25; yy = 30 - (i // 3) * 24; box(ax, xx, yy, 22, 16, t_, fc, fs=13)
for i in range(5):
    if i == 2: arrow(ax, (62, 30), (62, 22.2), cs="arc3,rad=0")
    elif i < 2: arrow(ax, (1 + i * 25 + 22.2, 38), (1 + (i + 1) * 25, 38))
    else: arrow(ax, (1 + (5 - i) * 25 + 0, 14), (1 + (5 - i) * 25 - 2.8, 14)) if False else None
arrow(ax, (51, 14), (48.2, 14)); arrow(ax, (26, 14), (23.2, 14))
a2 = fig.add_axes([0.68, 0.15, 0.3, 0.72]); x_ = np.arange(3); w = 0.36
a2.bar(x_ - w / 2, ld["before_dbfs"], w, color=C_ROSE, edgecolor=EDGE, label="before"); a2.bar(x_ + w / 2, ld["after_dbfs"], w, color=C_TEAL, edgecolor=EDGE, label="after")
a2.set_xticks(x_); a2.set_xticklabels(["clip 1", "clip 2", "clip 3"]); a2.set_ylabel("RMS level (dBFS)"); a2.legend(frameon=False, loc="lower right"); clean(a2); a2.set_ylim(-32, 0)
save(fig, "fig_11_6_voiceover")
print("ok", thr)
