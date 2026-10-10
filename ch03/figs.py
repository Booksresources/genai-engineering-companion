import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch03")
T2 = json.load(open("/home/claude/gb/ch3/toy2d.json"))
M = json.load(open("/home/claude/gb/ch3/metrics.json"))
GEN = {k: np.array(json.load(open(f"/home/claude/gb/ch3/gen_{k}.json"))["samples"]) for k in ["vae", "gan", "ar", "diffusion"]}
V = json.load(open("/home/claude/gb/ch3/gen_vae.json"))
from sklearn.datasets import load_digits
REAL = load_digits().data / 16
def tile(imgs, rows, cols):
    canvas_ = np.ones((rows * 9 + 1, cols * 9 + 1))
    for k in range(min(len(imgs), rows * cols)):
        r, c_ = divmod(k, cols); canvas_[1 + r * 9: 1 + r * 9 + 8, 1 + c_ * 9: 1 + c_ * 9 + 8] = 1 - np.clip(imgs[k].reshape(8, 8), 0, 1)
    return canvas_

# 3.1 temperature and sampling
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.4, 1]})
chars = [c for c, _ in T2["temp"]["1.0"]["top"]]
x = np.arange(len(chars)); w = 0.27
for k, (Tk, col) in enumerate([("0.5", L_TEAL), ("1.0", L_BLUE), ("1.5", L_RED)]):
    d = dict(T2["temp"][Tk]["top"]); vals = [d.get(ch, 0) for ch in chars]
    a1.bar(x + (k - 1) * w, vals, w, color=col, edgecolor=EDGE, label=f"T = {Tk} (entropy {T2['temp'][Tk]['entropy_bits']:.2f} bits)")
a1.set_xticks(x); a1.set_xticklabels([repr(ch) for ch in chars], fontsize=13); a1.set_ylabel("Probability of next character")
a1.set_title("(a) Temperature reshapes the distribution"); a1.legend(frameon=False, loc="upper right"); clean(a1)
sc = T2["sampling_check"]
a2.bar(np.arange(len(sc)) - 0.18, [s[1] for s in sc], 0.36, color=C_BLUE, edgecolor=EDGE, label="model probability")
a2.bar(np.arange(len(sc)) + 0.18, [s[2] for s in sc], 0.36, color=C_AMBER, edgecolor=EDGE, label="frequency in 2,000 samples")
a2.set_xticks(range(len(sc))); a2.set_xticklabels([repr(s[0]) for s in sc], fontsize=13); a2.set_title("(b) Sampling matches probabilities")
a2.legend(frameon=False, loc="upper right"); clean(a2)
fig.tight_layout(w_pad=3); save(fig, "fig_3_1_sampling")

# 3.2 VAE diagram
fig, ax = canvas(13, 5.6, 120, 50)
box(ax, 1, 18, 14, 14, "input image x", C_GREY, fs=13)
box(ax, 20, 16, 18, 18, "encoder network", C_LAV, fs=13)
box(ax, 44, 30, 14, 9, "mean μ", C_BLUE, fs=13); box(ax, 44, 11, 14, 9, "std σ", C_BLUE, fs=13)
box(ax, 64, 18, 20, 14, "z = μ + σ · ε\n(ε ~ noise)", C_AMBER, fs=13)
box(ax, 89, 16, 16, 18, "decoder network", C_LAV, fs=13)
box(ax, 108, 18, 11, 14, "x̂", C_GREY, fs=15)
arrow(ax, (15.2, 25), (20, 25)); arrow(ax, (38.2, 28), (44, 34)); arrow(ax, (38.2, 22), (44, 16)); arrow(ax, (58.2, 34), (64, 28)); arrow(ax, (58.2, 16), (64, 22))
arrow(ax, (84.2, 25), (89, 25)); arrow(ax, (105.2, 25), (108, 25))
label(ax, 60, 45.5, "Loss = reconstruction error + KL( q(z|x) ‖ N(0, I) )", fs=14, bold=True)
label(ax, 60, 4, "To generate: sample z ~ N(0, I) and run only the decoder.", fs=12.5, style="italic")
save(fig, "fig_3_2_vae")

# 3.3 VAE latent space (real)
lat = np.array(V["latent"]["mu"]); ly = np.array(V["latent"]["y"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.0))
cmap = plt.get_cmap("tab10")
for k in range(10):
    s = lat[ly == k]; a1.scatter(s[:, 0], s[:, 1], s=22, color=cmap(k), label=str(k))
    a1.text(np.median(s[:, 0]), np.median(s[:, 1]), str(k), fontsize=15, fontweight="bold", color=BLACK,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
a1.set_xlabel("latent dimension 1"); a1.set_ylabel("latent dimension 2"); a1.set_title("(a) Test digits encoded into a 2-D latent space"); clean(a1)
a2.imshow(tile(np.array(V["latent_grid"]), 12, 12), cmap="gray", vmin=0, vmax=1); a2.axis("off")
a2.set_title("(b) Decoding a grid of latent points")
fig.tight_layout(w_pad=3); save(fig, "fig_3_3_latent")

# 3.4 GAN diagram
fig, ax = canvas(13, 5.4, 120, 48)
box(ax, 1, 20, 14, 10, "noise z", C_GREY, fs=13); box(ax, 21, 18, 20, 14, "generator G", C_ROSE, fs=13)
box(ax, 47, 26, 16, 10, "fake sample", C_GREY, fs=12.5); box(ax, 47, 8, 16, 10, "real sample", C_GREEN, fs=12.5)
box(ax, 70, 14, 22, 16, "discriminator D: real or fake?", C_ROSE, fs=13)
arrow(ax, (15.2, 25), (21, 25)); arrow(ax, (41.2, 27), (47, 31)); arrow(ax, (63.2, 31), (70, 25)); arrow(ax, (63.2, 13), (70, 19))
box(ax, 97, 26, 22, 12, "D learns to tell them apart", C_GREY, fs=12, bold=False)
box(ax, 97, 8, 22, 12, "G learns to fool D", C_GREY, fs=12, bold=False)
arrow(ax, (92.2, 24), (97, 31)); arrow(ax, (97, 12), (31, 17.8), cs="arc3,rad=0.25", ls="--")
label(ax, 60, 45, "A two-player game: training alternates between D and G", fs=14, bold=True)
save(fig, "fig_3_4_gan")

# 3.5 GAN on 8 Gaussians (real)
g = T2["gan2d"]; cen = np.array(g["centers"])
fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
for ax, st in zip(axes, ["0", "500", "2000", "6000"]):
    p = np.array(g["snaps"][st]["pts"]); ax.scatter(p[:, 0], p[:, 1], s=4, color=L_RED, alpha=0.5)
    ax.scatter(cen[:, 0], cen[:, 1], s=90, marker="x", color=BLACK, lw=2)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title(f"step {st}: {g['snaps'][st]['modes']}/8 modes", fontsize=13.5)
fig.tight_layout(); save(fig, "fig_3_5_gan2d")

# 3.6 Autoregressive diagram
fig, ax = canvas(13, 5.6, 120, 50)
for r in range(4):
    for c_ in range(4):
        k = r * 4 + c_
        fc = C_BLUE if k < 6 else (C_AMBER if k == 6 else "white")
        ax.add_patch(FancyBboxPatch((4 + c_ * 7, 36 - r * 7), 6.4, 6.4, boxstyle="round,pad=0,rounding_size=0.3", fc=fc, ec=EDGE, lw=1.2))
        ax.text(7.2 + c_ * 7, 39.2 - r * 7, str(k + 1), ha="center", va="center", fontsize=11.5, color=BLACK)
label(ax, 17.5, 8, "Pixels in raster order:\nblue = known, amber = next", fs=12.5)
box(ax, 42, 20, 26, 14, "Transformer reads pixels 1–6", C_LAV, fs=13)
arrow(ax, (33, 27), (42, 27)); arrow(ax, (68.2, 27), (76, 27))
xs = np.arange(17); ps = np.exp(-0.5 * ((xs - 11) / 2.2) ** 2); ps /= ps.sum()
ax.bar(77 + xs * 2.4, ps * 120, 2.0, bottom=14, color=C_AMBER, edgecolor=EDGE, lw=0.6)
label(ax, 97, 10, "p(pixel 7 = 0 … 16 | pixels 1–6)", fs=12.5)
label(ax, 60, 46.5, "Sample pixel 7 from this distribution, append it, and repeat for all 64 pixels", fs=13.5, bold=True)
save(fig, "fig_3_6_ar")

# 3.7 Diffusion forward + flow matching (real)
fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
for ax, t in zip(axes[0], ["0", "10", "30", "99"]):
    p = np.array(T2["forward"][t]); ax.scatter(p[:, 0], p[:, 1], s=4, color=L_BLUE); ax.set_title(f"forward step {t}", fontsize=13)
for ax, k in zip(axes[1], ["1", "2", "4", "20"]):
    p = np.array(T2["flow"][k]["pts"]); ax.scatter(p[:, 0], p[:, 1], s=4, color=L_TEAL)
    ax.set_title(f"flow, {k} step{'s' if k != '1' else ''} (dist {T2['flow'][k]['nn_dist']:.3f})", fontsize=12.5)
for ax in axes.flat: ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
axes[0, 0].set_ylabel("Diffusion: data → noise", fontsize=13.5); axes[1, 0].set_ylabel("Flow matching: noise → data", fontsize=13.5)
fig.tight_layout(); save(fig, "fig_3_7_diffusion_flow")

# 3.8 flow paths
paths = np.array(T2["flow_paths"]); data = np.array(T2["data"])
fig, ax = plt.subplots(figsize=(7.5, 6.0))
ax.scatter(data[:, 0], data[:, 1], s=4, color="#BBBBBB", label="training data")
for i in range(paths.shape[1]): ax.plot(paths[:, i, 0], paths[:, i, 1], color=L_TEAL, lw=1.2)
ax.scatter(paths[0, :, 0], paths[0, :, 1], s=22, color=BLACK, label="start (noise)"); ax.scatter(paths[-1, :, 0], paths[-1, :, 1], s=22, color=L_RED, label="end (sample)")
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.legend(frameon=False, loc="lower left")
ax.set_title("Flow matching: 60 sample paths, 20 Euler steps")
save(fig, "fig_3_8_paths")

# 3.9 samples from four generators (real)
fig, axes = plt.subplots(1, 5, figsize=(13, 3.6))
rng = np.random.default_rng(0)
panels = [("real digits", REAL[rng.choice(len(REAL), 36, replace=False)])] + [(n, GEN[k][:36]) for n, k in [("VAE", "vae"), ("GAN", "gan"), ("autoregressive", "ar"), ("diffusion", "diffusion")]]
for ax, (n, S) in zip(axes, panels):
    ax.imshow(tile(S, 6, 6), cmap="gray", vmin=0, vmax=1); ax.set_title(n, fontsize=14); ax.axis("off")
fig.tight_layout(); save(fig, "fig_3_9_samples")

# 3.10 metric comparison (real)
names = ["vae", "gan", "ar", "diffusion"]; labels = ["VAE", "GAN", "Autoregressive", "Diffusion"]; cols = [C_LAV, C_ROSE, C_BLUE, C_AMBER]
fig, axes = plt.subplots(2, 2, figsize=(12, 8.4)); axes = axes.flatten()
specs = [("fd", "Fréchet distance (lower is better)", False), ("confidence", "Classifier confidence", False),
         ("class_entropy_bits", "Class diversity (bits; max 3.32)", False), ("sample_time_1000", "Seconds per 1,000 samples (log)", True)]
for ax, (key, title, logy) in zip(axes, specs):
    vals = [M[n][key] for n in names]
    ax.bar(range(4), vals, color=cols, edgecolor=EDGE)
    for i, v in enumerate(vals): ax.text(i, v * (1.15 if logy else 1) + (0 if logy else max(vals) * 0.02), f"{v:.2f}", ha="center", fontsize=11.5, fontweight="bold")
    if key in ("fd", "confidence", "class_entropy_bits"):
        ax.axhline(M["real (held-out test)"][key], color=BLACK, ls="--", lw=1.3)
    if logy:
        from matplotlib.ticker import FuncFormatter
        ax.set_yscale("log"); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); ax.set_ylim(0.005, 300)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=12.5); ax.set_title(title, fontsize=13.5); clean(ax)
axes[1].set_ylim(0.8, 1.0); axes[2].set_ylim(2.8, 3.45)
fig.text(0.5, -0.01, "Dashed line: real held-out digits", ha="center", fontsize=12.5, color=BLACK)
fig.tight_layout(w_pad=2, h_pad=2.5); save(fig, "fig_3_10_metrics")
print("ok")
