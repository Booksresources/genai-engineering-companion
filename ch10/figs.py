import sys, json, torch
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch10")
EV = json.load(open("t2i_eval.json")); DB = json.load(open("dreambooth.json"))
def im(ax, x, title=None, fs=11.5):
    ax.imshow(1 - (x[0].numpy() + 1) / 2, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=fs)
import textwrap

# 10.1 anatomy
fig, ax = canvas(13, 6.2, 124, 56)
box(ax, 1, 34, 20, 12, "prompt: \"a TrailMug on a mossy rock at dawn\"", C_GREY, fs=12.5, bold=False)
box(ax, 26, 34, 20, 12, "text encoder (CLIP / T5)", C_LAV, fs=13.5)
box(ax, 1, 10, 20, 12, "random latent noise (seed)", C_GREY, fs=13)
box(ax, 52, 18, 30, 22, "denoiser (U-Net or DiT)\n× 20–50 steps\nwith classifier-free guidance", C_BLUE, fs=13.5)
box(ax, 88, 22, 16, 14, "VAE decoder", C_TEAL, fs=14); box(ax, 108, 22, 15, 14, "image", C_GREEN, fs=14)
arrow(ax, (21.2, 40), (26, 40)); arrow(ax, (46.2, 40), (52, 34)); arrow(ax, (21.2, 16), (52, 24)); arrow(ax, (82.2, 29), (88, 29)); arrow(ax, (104.2, 29), (108, 29))
label(ax, 67, 49, "scheduler / sampler, steps, guidance, resolution", fs=13.5, style="italic")
save(fig, "fig_10_1_anatomy")

# 10.2 mini t2i samples + adherence
sm = torch.load("t2i_samples.pt"); gw = EV["guidance"]
fig = plt.figure(figsize=(13, 6.4))
for k in range(6):
    ax = fig.add_axes([0.01 + (k % 3) * 0.155, 0.53 - (k // 3) * 0.47, 0.14, 0.33])
    p_ = sm["prompts"][k].split(); short = f"{p_[1]}, {'glasses' if p_[4] == 'with' else 'no glasses'},\n{p_[7]} bg"; im(ax, sm["S"][k], short, fs=12)
a = fig.add_axes([0.55, 0.13, 0.43, 0.75])
ws = list(gw.keys())
for key, col, mk in [("expression", L_BLUE, "o"), ("glasses", L_AMBER, "s"), ("background", L_TEAL, "^"), ("all", L_RED, "D")]:
    a.plot([float(w) for w in ws], [100 * gw[w][key] for w in ws], marker=mk, ms=8, lw=2.6, color=col, label=key if key != "all" else "all three correct")
a.set_xlabel("guidance scale"); a.set_ylabel("Matches the prompt (%)"); a.set_ylim(0, 105); a.legend(frameon=False, loc="lower left", fontsize=12); a.grid(True, color="#DDD"); clean(a)
save(fig, "fig_10_2_t2i")

# 10.3 control: inpainting and img2img (real)
ip = torch.load("inpaint.pt"); ii = torch.load("img2img.pt")
fig, axes = plt.subplots(2, 5, figsize=(13, 5.8))
for k in range(2):
    im(axes[0, 2 * k], ip["known"][k], "original"); im(axes[0, 2 * k + 1], ip["out"][k], "inpaint lower half:\n\"smiling\"")
axes[0, 4].axis("off"); axes[0, 4].text(0.0, 0.5, f"{100*EV['inpaint']['smiling_after']:.0f}% now smiling;\nupper half kept\nexactly", fontsize=13.5, va="center", color=BLACK)
im(axes[1, 0], ii["init"][1], "input")
for k, s_ in enumerate(["0.3", "0.5", "0.7", "1.0"]):
    im(axes[1, k + 1], ii[s_][1], f"strength {s_}\ngrey bg {100*EV['img2img'][s_]['grey_background']:.0f}%")
fig.tight_layout(); save(fig, "fig_10_3_control")

# 10.4 controlnet / ip-adapter concept
fig, ax = canvas(13, 5.6, 124, 50)
box(ax, 1, 28, 22, 15, "control image (edges, depth, pose)", C_AMBER, fs=13)
box(ax, 28, 30, 26, 12, "ControlNet: trainable copy of the encoder", C_ROSE, fs=13)
box(ax, 1, 8, 22, 12, "reference image (style or subject)", C_LAV, fs=13)
box(ax, 28, 8, 26, 12, "image prompt adapter: image features into cross-attention", C_LAV, fs=12.5)
box(ax, 62, 16, 34, 18, "frozen base denoiser (text-conditioned)", C_BLUE, fs=14)
box(ax, 102, 18, 20, 14, "controlled image", C_GREEN, fs=13.5)
arrow(ax, (23.2, 36), (28, 36)); arrow(ax, (23.2, 14), (28, 14)); arrow(ax, (54.2, 36), (62, 29)); arrow(ax, (54.2, 14), (62, 21)); arrow(ax, (96.2, 25), (102, 25))
save(fig, "fig_10_4_controlnet")

# 10.5 dreambooth (real)
refs = torch.load("db_refs.pt"); d0 = torch.load("db_False.pt"); d1 = torch.load("db_True.pt")
fig = plt.figure(figsize=(13, 6.2))
for k in range(3): im(fig.add_axes([0.01 + k * 0.1, 0.62, 0.09, 0.28]), refs[k], "reference" if k == 0 else None)
fig.text(0.16, 0.95, "6 reference images of \"sks\" (a hat)", ha="center", fontsize=13.5, fontweight="bold", color=BLACK)
for r, (d, lab) in enumerate([(d0, "no prior preservation"), (d1, "with prior preservation")]):
    fig.text(0.21, 0.45 - r * 0.24, lab + ":  left = \"sks\" prompt, right = plain", ha="center", fontsize=12, fontweight="bold", color=BLACK)
    for k in range(2):
        im(fig.add_axes([0.01 + k * 0.1, 0.24 - r * 0.24 + 0.0, 0.09, 0.18]), d["new"][k])
        im(fig.add_axes([0.21 + k * 0.1, 0.24 - r * 0.24 + 0.0, 0.09, 0.18]), d["old"][k])
a = fig.add_axes([0.5, 0.15, 0.48, 0.75])
keys = ["before fine-tuning", "without prior preservation", "with prior preservation (1:4)", "with prior preservation (1:1)"]
labs = ["before", "no prior\npreservation", "prior\n(1:4)", "prior\n(1:1)"]
x = np.arange(4); w = 0.36
a.bar(x - w / 2, [100 * DB[k]["hat_rate_new_prompts"] for k in keys], w, color=C_TEAL, edgecolor=EDGE, label="hat with \"sks\" prompts (want high)")
a.bar(x + w / 2, [100 * DB[k]["hat_leak_old_prompts"] for k in keys], w, color=C_ROSE, edgecolor=EDGE, label="hat leaks into plain prompts (want low)")
a.set_xticks(x); a.set_xticklabels(labs, fontsize=12.5); a.set_ylabel("Generated faces with a hat (%)"); a.set_ylim(0, 110)
a.legend(frameon=False, loc="upper right", fontsize=11.5); clean(a)
save(fig, "fig_10_5_dreambooth")

# 10.6 campaign consistency + Muse
fig, ax = canvas(13, 6.0, 124, 54)
st = [("brief + brand rules", C_GREY), ("prompt template + product LoRA + reference image", C_LAV), ("generate 4–8 candidates (fixed seeds logged)", C_BLUE),
      ("automatic checks: safety, product match, artifacts", C_AMBER), ("upscale and finish", C_TEAL), ("human approval + provenance label", C_GREEN)]
for i, (t, fc) in enumerate(st):
    x = 1 + i * 20.5; box(ax, x, 18, 18.5, 20, t, fc, fs=12.5)
    if i < 5: arrow(ax, (x + 18.7, 28), (x + 20.5, 28))
label(ax, 62, 48, "Muse product imagery workflow", fs=15, bold=True)
label(ax, 62, 7, "every image logs prompt, model, LoRA, seed, sampler, steps, guidance, and reviewer", fs=13, style="italic")
save(fig, "fig_10_6_workflow")
print("ok")
