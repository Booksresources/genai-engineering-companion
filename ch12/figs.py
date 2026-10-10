import sys, json, torch
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
from video import clips
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch12")
E = json.load(open("video_eval.json")); iv = torch.load("i2v.pt")
rows = [("real", clips(4, seed=7)[0]), ("per-frame image model", torch.load("samples_frame.pt")[0]), ("space-time video model", torch.load("samples_video.pt")[0]),
        ("image-to-video (frame 1 given)", iv["gen"][0])]
fig, axes = plt.subplots(4, 8, figsize=(13, 7.0))
for r, (lab, V) in enumerate(rows):
    for t in range(8):
        ax = axes[r, t]; ax.imshow(1 - (V[0, t].numpy() + 1) / 2, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(f"t = {t+1}", fontsize=13)
    axes[r, 0].set_ylabel("\n".join(__import__("textwrap").wrap(lab, 14)), fontsize=12.5, rotation=0, ha="right", va="center")
fig.tight_layout(); save(fig, "fig_12_1_filmstrips")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0))
keys = [("real", "real"), ("frame", "per-\nframe"), ("video", "space-\ntime"), ("image_to_video", "image-\nto-video")]
cols = [C_GREY, C_ROSE, C_BLUE, C_TEAL]
v1 = [E[k]["mean_jump_px"] for k, _ in keys]; a1.bar(range(4), v1, color=cols, edgecolor=EDGE)
for i, v in enumerate(v1): a1.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=13.5, fontweight="bold")
a1.set_xticks(range(4)); a1.set_xticklabels([l for _, l in keys], fontsize=12.5); a1.set_ylim(0, 3.3); a1.set_ylabel("Mean jump (pixels)"); a1.set_title("(a) Temporal consistency"); clean(a1)
v2 = [100 * E[k]["one_object_every_frame"] for k, _ in keys]; a2.bar(range(4), v2, color=cols, edgecolor=EDGE)
for i, v in enumerate(v2): a2.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=13.5, fontweight="bold")
a2.set_xticks(range(4)); a2.set_xticklabels([l for _, l in keys], fontsize=12.5); a2.set_ylim(0, 115); a2.set_ylabel("Clean clips (%)"); a2.set_title("(b) Per-frame correctness"); clean(a2)
fig.tight_layout(w_pad=3); save(fig, "fig_12_2_metrics")

fig, ax = canvas(13, 6.2, 124, 56)
label(ax, 62, 53, "From image to video denoisers", fs=15, bold=True)
opts = [("Per-frame image model", "each frame denoised alone: no motion, no consistency", C_ROSE),
        ("Inflated U-Net", "image U-Net + temporal convolution / attention layers inserted between spatial ones", C_AMBER),
        ("Space-time latent transformer", "video VAE compresses space and time; patches over (t, h, w) become tokens; full or factorized attention", C_BLUE)]
for i, (t, d, fc) in enumerate(opts):
    y = 36 - i * 15; box(ax, 2, y, 30, 12, t, fc, fs=14); box(ax, 36, y, 86, 12, d, "white", fs=13.5, bold=False)
save(fig, "fig_12_3_architectures")

from matplotlib.ticker import FuncFormatter
dur = np.array([2, 4, 8, 16, 30, 60]); fig, ax = plt.subplots(figsize=(11, 5.4))
for (w, h, lab), col in zip([(854, 480, "480p"), (1280, 720, "720p"), (1920, 1080, "1080p")], [L_TEAL, L_BLUE, L_RED]):
    tokens = (w / 16) * (h / 16) * (dur * 24 / 4)
    ax.plot(dur, tokens, "-o", lw=2.6, ms=8, color=col, label=lab)
ax.axhline(4096, color=BLACK, ls="--", lw=1.3); ax.text(2.1, 5200, "one 1024×1024 image ≈ 4,096 tokens", fontsize=12.5)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(dur); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}")); ax.set_xlabel("Clip length (seconds at 24 fps)"); ax.set_ylabel("Latent tokens per clip (log)")
ax.set_title("Assumes 8× spatial, 4× temporal compression, 2×2 patches", fontsize=13); ax.legend(frameon=False); ax.grid(True, which="major", color="#DDD"); clean(ax)
save(fig, "fig_12_4_tokens")

fig, ax = canvas(13, 5.8, 124, 52)
items = [("Per-frame quality", "image metrics on frames; artifacts", C_BLUE), ("Temporal consistency", "flicker, identity drift, object permanence", C_AMBER),
         ("Motion", "amount, smoothness, physical plausibility", C_TEAL), ("Prompt alignment", "text–video similarity; requested actions happen", C_LAV),
         ("Distribution", "Fréchet video distance (FVD)", C_GREY), ("Human preference", "side-by-side ratings; task success", C_GREEN)]
for i, (t, d, fc) in enumerate(items):
    x = 2 + (i % 3) * 40.5; y = 27 - (i // 3) * 22
    box(ax, x, y + 9, 38, 9, t, fc, fs=14); box(ax, x, y, 38, 8.5, d, "white", fs=12.5, bold=False)
label(ax, 62, 49.5, "What to measure in generated video", fs=15, bold=True)
save(fig, "fig_12_5_evaluation")
print("ok")
