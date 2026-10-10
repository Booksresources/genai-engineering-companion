import sys, json, torch
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
from diff9 import digits, faces, schedule
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch09")
E = json.load(open("eval9.json")); FE = json.load(open("faces_eval.json"))
def tile(S, rows, cols, size):
    S = ((S.view(-1, size, size).numpy() + 1) / 2); c_ = np.ones((rows * (size + 1) + 1, cols * (size + 1) + 1))
    for k in range(min(len(S), rows * cols)):
        r, q = divmod(k, cols); c_[1 + r * (size + 1): 1 + r * (size + 1) + size, 1 + q * (size + 1): 1 + q * (size + 1) + size] = 1 - S[k]
    return c_

# 9.1 forward process
X, Y = digits(); Fc, Lc = faces(10); betas, abar = schedule("cosine", 200); g = torch.Generator().manual_seed(0)
ts = [0, 20, 50, 100, 150, 199]
fig, axes = plt.subplots(2, 6, figsize=(13, 5.0))
for row, (img, size) in enumerate([(X[0], 8), (Fc[2], 16)]):
    e = torch.randn(img.shape, generator=g)
    for ax, t in zip(axes[row], ts):
        a = abar[t]; xt = a.sqrt() * img + (1 - a).sqrt() * e
        ax.imshow(1 - (xt[0].numpy() + 1) / 2, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        if row == 0: ax.set_title(f"t = {t}", fontsize=14)
axes[0, 0].set_ylabel("digit", fontsize=14); axes[1, 0].set_ylabel("face", fontsize=14)
fig.suptitle("Forward process: noise is added until nothing but noise remains (cosine schedule, T = 200)", fontsize=14.5, fontweight="bold", color=BLACK)
fig.tight_layout(); save(fig, "fig_9_1_forward")

# 9.2 schedules
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0))
for kind, col in [("linear", L_RED), ("cosine", L_BLUE)]:
    b, a = schedule(kind, 200); a1.plot(range(200), a.numpy(), lw=2.8, color=col, label=kind)
    snr = (a / (1 - a)).numpy(); a2.plot(range(200), np.log10(snr), lw=2.8, color=col, label=kind)
a1.set_xlabel("step t"); a1.set_ylabel("ᾱ_t (signal kept)"); a1.legend(frameon=False); a1.grid(True, color="#DDD"); clean(a1)
a2.set_xlabel("step t"); a2.set_ylabel("log₁₀ SNR"); a2.legend(frameon=False); a2.grid(True, color="#DDD"); clean(a2)
a1.set_title("(a) Signal kept"); a2.set_title("(b) Signal-to-noise")
fig.tight_layout(w_pad=3); save(fig, "fig_9_2_schedules")

# 9.3 samplers (real)
sm = E["samplers"]; names = list(sm.keys())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
a1.bar(range(len(names)), [sm[n]["fd"] for n in names], color=[C_ROSE] + [C_BLUE] * 4, edgecolor=EDGE)
for i, n in enumerate(names): a1.text(i, sm[n]["fd"] + 0.12, f"{sm[n]['fd']:.2f}", ha="center", fontsize=13, fontweight="bold")
a1.set_xticks(range(len(names))); a1.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=13); a1.set_ylabel("Fréchet distance (lower = better)"); a1.set_title("(a) Quality"); clean(a1)
a2.bar(range(len(names)), [sm[n]["seconds"] for n in names], color=[C_ROSE] + [C_BLUE] * 4, edgecolor=EDGE)
for i, n in enumerate(names): a2.text(i, sm[n]["seconds"] + 0.3, f"{sm[n]['seconds']:.1f} s", ha="center", fontsize=13, fontweight="bold")
a2.set_xticks(range(len(names))); a2.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=13); a2.set_ylabel("Seconds per 500 samples"); a2.set_title("(b) Time"); clean(a2)
fig.tight_layout(w_pad=3); save(fig, "fig_9_3_samplers")

# 9.4 U-Net and DiT diagrams
fig, ax = canvas(13, 6.6, 120, 60)
label(ax, 28, 57, "U-Net", fs=16, bold=True); label(ax, 90, 57, "Diffusion transformer (DiT)", fs=16, bold=True)
blocks = [(2, 40, 18, 9, "8×8, 16 ch", C_BLUE), (12, 24, 18, 9, "4×4, 32 ch", C_TEAL), (34, 40, 18, 9, "8×8, 16 ch", C_BLUE)]
for x, y, w, h, t, fc in blocks: box(ax, x, y, w, h, t, fc, fs=14)
arrow(ax, (11, 40), (17, 33.2)); arrow(ax, (26, 33.2), (40, 40)); arrow(ax, (20.2, 44.5), (34, 44.5), ls="--"); label(ax, 27, 47.5, "skip", fs=13)
box(ax, 2, 6, 50, 12, "time + class embedding added in every block", C_AMBER, fs=14)
toks = [(66 + i * 7, 40) for i in range(6)]
for x, y in toks: ax.add_patch(FancyBboxPatch((x, y), 5.5, 7, boxstyle="round,pad=0,rounding_size=0.3", fc=C_LAV, ec=EDGE))
label(ax, 87, 50, "image → patches → tokens", fs=14)
box(ax, 66, 22, 41, 12, "Transformer blocks with adaptive layer norm", C_BLUE, fs=14)
arrow(ax, (86, 40), (86, 34.2)); box(ax, 66, 6, 41, 11, "unpatchify → predicted noise", C_TEAL, fs=14); arrow(ax, (86, 22), (86, 17.2))
box(ax, 109, 22, 10, 12, "t, y", C_AMBER, fs=14); arrow(ax, (109, 28), (107.2, 28))
save(fig, "fig_9_4_architectures")

# 9.5 CFG (real)
gw = E["guidance"]; ws = list(gw.keys())
fig = plt.figure(figsize=(13, 5.6))
a1 = fig.add_axes([0.07, 0.15, 0.30, 0.75])
a1.plot([float(w) for w in ws], [100 * gw[w]["class_match"] for w in ws], "-o", lw=2.6, ms=9, color=L_BLUE, label="matches requested digit (%)")
a1b = a1.twinx(); a1b.plot([float(w) for w in ws], [gw[w]["fd"] for w in ws], "--s", lw=2.6, ms=9, color=L_RED, label="Fréchet distance")
a1.set_xlabel("guidance scale w"); a1.set_ylabel("Digit match (%)"); a1b.set_ylabel("Fréchet distance")
a1.set_ylim(40, 100); h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a1b.get_legend_handles_labels(); a1.legend(h1 + h2, l1 + l2, frameon=False, loc="lower left", fontsize=12)
for k, (w, pos) in enumerate([("1.0", 0.50), ("3.0", 0.76)]):
    ax = fig.add_axes([pos, 0.08, 0.23, 0.82]); S = torch.load(f"samples_w{w}.pt")
    ax.imshow(tile(S, 4, 10, 8), cmap="gray", vmin=0, vmax=1); ax.axis("off"); ax.set_title(f"w = {float(w):g}", fontsize=14)
save(fig, "fig_9_5_cfg")

# 9.6 latent diffusion
ae = torch.load("ae_recon.pt")
fig = plt.figure(figsize=(13, 5.4))
ax = fig.add_axes([0, 0.0, 0.6, 1]); ax.set_xlim(0, 72); ax.set_ylim(0, 50); ax.axis("off")
box(ax, 1, 20, 12, 12, "image\n16×16", C_GREY, fs=14); box(ax, 17, 20, 12, 12, "encoder", C_LAV, fs=14)
box(ax, 33, 20, 14, 12, "latent 4×4×4", C_AMBER, fs=14); box(ax, 51, 20, 12, 12, "decoder", C_LAV, fs=14)
arrow(ax, (13.2, 26), (17, 26)); arrow(ax, (29.2, 26), (33, 26)); arrow(ax, (47.2, 26), (51, 26)); arrow(ax, (63.2, 26), (67, 26))
box(ax, 28, 38, 24, 9, "diffusion runs here", C_TEAL, fs=14); arrow(ax, (40, 38), (40, 32.2))
label(ax, 36, 9, f"4× fewer numbers; test reconstruction MSE {FE['autoencoder']['test_mse']:.3f}", fs=13.5)
a2 = fig.add_axes([0.62, 0.1, 0.36, 0.8])
a2.imshow(np.vstack([tile(ae["orig"], 1, 8, 16), tile(ae["rec"], 1, 8, 16)]), cmap="gray", vmin=0, vmax=1); a2.axis("off")
a2.set_title("top: original   bottom: reconstructed", fontsize=13.5)
save(fig, "fig_9_6_latent")

# 9.7 lab: noise to faces (real)
tr = torch.load("trajectory.pt")
fig, axes = plt.subplots(3, 7, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1] * 6 + [2.4]})
names = ["frown", "neutral", "smile"]; steps_lab = ["noise", "step 4", "step 8", "step 12", "step 16", "final"]
for r in range(3):
    for k in range(6):
        ax = axes[r, k]; ax.imshow(1 - (tr[k, r, 0].numpy() + 1) / 2, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(steps_lab[k], fontsize=13)
    axes[r, 0].set_ylabel(names[r], fontsize=14)
gs = axes[0, 6].get_gridspec()
for a in axes[:, 6]: a.remove()
axb = fig.add_subplot(gs[:, 6]); S = torch.load("faces_w3.0.pt")
axb.imshow(tile(S, 8, 3, 16), cmap="gray", vmin=0, vmax=1); axb.axis("off"); axb.set_title("samples (columns:\nfrown, neutral, smile)", fontsize=13)
fig.tight_layout(); save(fig, "fig_9_7_faces")

# 9.8 U-Net vs DiT samples (real)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, nm in zip(axes, ["U-Net", "DiT"]):
    S = torch.load(f"samples_{nm}.pt"); ax.imshow(tile(S, 4, 10, 8), cmap="gray", vmin=0, vmax=1); ax.axis("off")
    ax.set_title(f"{nm}: FD {E['arch'][nm]['fd']:.1f}, digit match {100*E['arch'][nm]['class_match']:.0f}%", fontsize=14)
fig.tight_layout(); save(fig, "fig_9_8_unet_dit")
print("ok")

# 9.x training curves (real)
fig, ax = plt.subplots(figsize=(11, 5.4))
for f, col, lab in [("unet_train.json", L_BLUE, "U-Net, digits"), ("dit_train.json", L_RED, "DiT, digits"), ("faces_train.json", L_TEAL, "U-Net, faces")]:
    lg = np.array(json.load(open(f))["log"]); k = 3
    ax.plot(lg[:, 0], np.convolve(lg[:, 1], np.ones(k) / k, mode="same"), lw=2.6, color=col, label=lab)
ax.set_xlabel("Training step"); ax.set_ylabel("Noise-prediction loss (MSE)"); ax.set_ylim(0, 0.6)
ax.legend(frameon=False); ax.grid(True, color="#DDD"); clean(ax)
save(fig, "fig_9_9_training")
