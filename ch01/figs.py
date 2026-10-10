import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch01")
D = json.load(open("/home/claude/gb/ch1/demo.json"))

# 1.1 Generative AI in one picture
fig, ax = canvas(13, 7.2, 120, 66)
label(ax, 60, 63, "Generative AI in one picture: any input, any output", fs=15, bold=True)
ins = [("Text prompt", C_GREY), ("Image or sketch", C_GREY), ("Audio or voice", C_GREY), ("Documents and data", C_GREY)]
outs = [("Text and code", C_BLUE, "chat, writing, coding, extraction"), ("Images", C_AMBER, "text-to-image, editing, inpainting"),
        ("Audio", C_TEAL, "speech, voice, music"), ("Video", C_ROSE, "text-to-video, image-to-video"), ("Actions", C_LAV, "tool calls, agents")]
for i, (t, fc) in enumerate(ins):
    box(ax, 1, 48 - i * 13, 22, 9, t, fc, fs=13)
    arrow(ax, (23.2, 52.5 - i * 13), (38, 33))
box(ax, 38, 21, 34, 24, "Foundation model\n(language, diffusion, or multimodal)", C_GREEN, fs=14)
for i, (t, fc, ex) in enumerate(outs):
    y = 53 - i * 12.2
    box(ax, 84, y, 15, 9, t, fc, fs=13)
    label(ax, 100.5, y + 4.5, ex, fs=11.5, ha="left", width=20)
    arrow(ax, (72.2, 33), (84, y + 4.5))
label(ax, 60, 2.5, "Multimodal models accept and produce several of these at once.", fs=12.5, style="italic")
save(fig, "fig_1_1_one_picture")

# 1.2 Five model families
fig, ax = canvas(13, 8.4, 120, 78)
fams = [("Autoregressive", "Predict the next token, append it, repeat.", C_BLUE),
        ("Diffusion", "Start from noise; remove a little noise per step.", C_AMBER),
        ("Flow matching", "Learn a velocity field that moves noise straight to data.", C_TEAL),
        ("VAE", "Encode to a latent code; decode it back; sample new codes.", C_LAV),
        ("GAN", "A generator tries to fool a discriminator.", C_ROSE)]
for i, (t, d, fc) in enumerate(fams):
    y = 64 - i * 15
    box(ax, 1, y, 20, 11, t, fc, fs=14)
    label(ax, 23, y + 5.5, d, fs=12.5, ha="left", width=30)
    x0 = 66
    if t == "Autoregressive":
        for k, tok in enumerate(["The", "cat", "sat", "on", "?"]):
            box(ax, x0 + k * 10, y + 2, 8.5, 7, tok, C_BLUE if tok != "?" else "white", fs=12)
        arrow(ax, (x0 + 44, y + 5.5), (x0 + 51, y + 5.5)); label(ax, x0 + 52, y + 5.5, "\"the\"", fs=12.5, ha="left")
    elif t == "Diffusion":
        for k, a in enumerate([1.0, 0.75, 0.5, 0.25, 0.0]):
            rng = np.random.default_rng(k)
            img = (1 - a) * np.outer(np.hanning(16), np.hanning(16)) + a * rng.random((16, 16))
            ax.imshow(img, extent=(x0 + k * 11, x0 + k * 11 + 8.5, y + 1.5, y + 9.5), cmap="Greys", aspect="auto", zorder=3)
            if k < 4: arrow(ax, (x0 + k * 11 + 8.7, y + 5.5), (x0 + k * 11 + 10.8, y + 5.5), ms=11)
    elif t == "Flow matching":
        ax.scatter([x0 + 3], [y + 5.5], s=200, color=C_GREY, edgecolor=EDGE, zorder=3); label(ax, x0 + 3, y + 0.8, "noise", fs=11.5)
        ax.scatter([x0 + 47], [y + 5.5], s=200, color=C_TEAL, edgecolor=EDGE, zorder=3); label(ax, x0 + 47, y + 0.8, "data", fs=11.5)
        arrow(ax, (x0 + 5, y + 5.5), (x0 + 45, y + 5.5), lw=2.2); label(ax, x0 + 25, y + 8.5, "straight path", fs=11.5)
    elif t == "VAE":
        box(ax, x0, y + 1.5, 11, 8, "encoder", C_LAV, fs=11.5); box(ax, x0 + 18, y + 2.5, 8, 6, "z", "white", fs=13)
        box(ax, x0 + 33, y + 1.5, 11, 8, "decoder", C_LAV, fs=11.5)
        arrow(ax, (x0 + 11.2, y + 5.5), (x0 + 18, y + 5.5)); arrow(ax, (x0 + 26.2, y + 5.5), (x0 + 33, y + 5.5))
    else:
        box(ax, x0, y + 1.5, 13, 8, "generator", C_ROSE, fs=11.5); box(ax, x0 + 30, y + 1.5, 16, 8, "discriminator", C_ROSE, fs=11.5)
        arrow(ax, (x0 + 13.2, y + 5.5), (x0 + 30, y + 5.5)); label(ax, x0 + 21.5, y + 8.2, "fake?", fs=11.5)
        arrow(ax, (x0 + 38, y + 1.5), (x0 + 6.5, y + 1.5), cs="arc3,rad=0.35", ls="--", lw=1.2)
save(fig, "fig_1_2_families")

# 1.3 Timeline
fig, ax = plt.subplots(figsize=(13, 5.8)); fig.subplots_adjust(left=0.02, right=0.98)
ax.set_xlim(2012.6, 2025.6); ax.set_ylim(-7, 7); ax.axis("off")
ax.plot([2013, 2025.3], [0, 0], color=BLACK, lw=2.2)
ev = [(2013.9, "VAE", 1.8), (2014.5, "GAN", -1.8), (2016.4, "PixelCNN,\nWaveNet", 3.6), (2017.5, "Transformer", -3.6), (2019.1, "GPT-2", 1.8),
      (2020.5, "DDPM\n(diffusion)", -1.8), (2021.05, "CLIP,\nDALL·E", 4.4), (2022.6, "Stable Diffusion,\nWhisper", -4.6), (2022.9, "ChatGPT", 2.0),
      (2023.2, "GPT-4\n(multimodal)", 5.2), (2024.1, "Sora preview\n(video)", -2.4), (2024.7, "Reasoning\nmodels", 2.8)]
for x, t, yv in ev:
    ax.plot([x, x], [0, yv * 0.8], color="#666", lw=1); ax.scatter(x, 0, s=60, color=L_BLUE, zorder=5)
    ax.text(x, yv, t, ha="center", va="center", fontsize=12, fontweight="bold", color=BLACK,
            bbox=dict(fc="white", ec=EDGE, boxstyle="round,pad=0.35", lw=1))
for y in range(2013, 2026, 2): ax.text(y, -0.55, str(y), ha="center", va="top", fontsize=11.5, color=BLACK)
save(fig, "fig_1_3_timeline")

# 1.4 Build spectrum
fig, ax = canvas(13, 6.4, 120, 58)
steps = [("Prompt", "Change the instructions and examples", "minutes · $", C_BLUE),
         ("Retrieve (RAG)", "Give the model your documents at query time", "days · $$", C_TEAL),
         ("Fine-tune", "Adjust weights for behavior, style, or a skill", "weeks · $$$", C_AMBER),
         ("Train", "Build a model from scratch", "months · $$$$$", C_ROSE)]
for i, (t, d, cst, fc) in enumerate(steps):
    x = 2 + i * 29.5; y = 12 + i * 8
    box(ax, x, y, 26, 12, t, fc, fs=14)
    label(ax, x + 13, y - 3.5, cst, fs=12.5, bold=True)
    label(ax, x + 13, y + 17, d, fs=12, width=24)
arrow(ax, (2, 4), (118, 4), lw=2.4)
label(ax, 60, 1, "More effort, cost, and control  →", fs=13, bold=True)
label(ax, 4, 54, "Start at the left; move right only when evaluations show you must.", fs=13, ha="left", style="italic")
save(fig, "fig_1_4_spectrum")

# 1.5 Lifecycle
fig, ax = canvas(13, 6.8, 120, 62)
st = [("1. Define the task and success metric", C_GREY), ("2. Prototype with prompts", C_BLUE), ("3. Build an evaluation set", C_TEAL),
      ("4. Improve: retrieval, tools, fine-tuning", C_AMBER), ("5. Harden: safety, cost, latency", C_ROSE), ("6. Deploy in stages", C_LAV), ("7. Monitor and learn", C_GREEN)]
cx, cy, R = 60, 30, 22
ang = np.linspace(90, 90 - 360, len(st) + 1)[:-1]
pos = []
for (t, fc), a in zip(st, ang):
    x = cx + 2.1 * R * np.cos(np.deg2rad(a)); y = cy + R * np.sin(np.deg2rad(a))
    box(ax, x - 12, y - 4.5, 24, 9, t, fc, fs=12); pos.append((x, y))
for k in range(len(pos)):
    (x1, y1), (x2, y2) = pos[k], pos[(k + 1) % len(pos)]
    arrow(ax, (x1 + (x2 - x1) * 0.28, y1 + (y2 - y1) * 0.28), (x1 + (x2 - x1) * 0.72, y1 + (y2 - y1) * 0.72), lw=1.6)
label(ax, cx, cy, "Evaluation\nat every step", fs=14, bold=True)
save(fig, "fig_1_5_lifecycle")

# 1.6 Same input, different outputs (real samples)
fig, ax = canvas(13, 6.6, 120, 60)
label(ax, 1, 57, "Prompt: \"ROMEO:\"   Model: the tiny Shakespeare GPT from the companion code", fs=13, bold=True, ha="left")
rows = [("greedy (temperature 0)", D["greedy"])] + [(f"sample {i+1} (temperature 0.8)", s) for i, s in enumerate(D["samples"])]
for i, (lab, s) in enumerate(rows):
    y = 45 - i * 10
    box(ax, 1, y, 26, 8, lab, C_GREY if i == 0 else C_BLUE, fs=12)
    ax.add_patch(FancyBboxPatch((29, y), 90, 8, boxstyle="round,pad=0,rounding_size=0.6", fc="white", ec=EDGE, lw=1))
    t, f = wrap_text(ax, s.replace("\n", " ⏎ ")[:110], 88, 8, 11.5)
    ax.text(30.2, y + 4, t, va="center", fontsize=f, family="DejaVu Sans Mono", color=BLACK)
label(ax, 60, 1.5, "Same input, different outputs: generative systems are probabilistic by design.", fs=12.5, style="italic")
save(fig, "fig_1_6_samples")

# 1.7 Cost model (illustrative prices)
req = np.logspace(3, 6, 50)
tiers = [("small model ($0.10 / $0.40 per M tokens)", 0.10, 0.40, L_TEAL), ("mid-size model ($1 / $4)", 1, 4, L_BLUE), ("large model ($5 / $15)", 5, 15, L_RED)]
fig, ax = plt.subplots(figsize=(10.5, 5.6))
for name, pin, pout, col in tiers:
    monthly = req * 30 * (1500 * pin + 300 * pout) / 1e6
    ax.plot(req, monthly, lw=2.8, color=col, label=name)
    ax.annotate(f"${monthly[-1]:,.0f}", (req[-1], monthly[-1]), xytext=(6, 0), textcoords="offset points", fontsize=12, fontweight="bold", va="center")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Requests per day"); ax.set_ylabel("Monthly API cost (USD, log)")
from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}")); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"${v:,.0f}"))
ax.set_title("1,500 input + 300 output tokens per request (illustrative prices)")
ax.legend(frameon=False, loc="upper left"); ax.grid(True, which="major", color="#DDD"); clean(ax); ax.set_xlim(1e3, 2.2e6)
save(fig, "fig_1_7_cost")

# 1.8 Muse architecture and roadmap
fig, ax = canvas(13, 7.4, 120, 68)
label(ax, 60, 65.5, "Muse: the running project and where each part is built", fs=15, bold=True)
box(ax, 2, 48, 22, 10, "Web app and API (Ch. 17, 23)", C_GREY, fs=12.5)
box(ax, 30, 48, 30, 10, "Orchestrator and agents (Ch. 15–16)", C_LAV, fs=12.5)
box(ax, 66, 48, 25, 10, "Safety and provenance layer (Ch. 20)", C_ROSE, fs=12.5)
box(ax, 96, 48, 22, 10, "Evaluation and monitoring (Ch. 19, 24)", C_GREEN, fs=12.5)
caps = [("On-brand copy: fine-tuned LLM (Ch. 5–8)", C_BLUE), ("Brand-guideline Q&A: RAG (Ch. 14)", C_TEAL), ("Product images: diffusion (Ch. 9–10)", C_AMBER),
        ("Voiceovers: TTS (Ch. 11)", C_TEAL), ("Short clips: video (Ch. 12)", C_ROSE), ("Brief and mood-board understanding (Ch. 13)", C_LAV)]
for i, (t, fc) in enumerate(caps):
    x = 2 + (i % 3) * 39; y = 26 - (i // 3) * 13
    box(ax, x, y, 36, 10, t, fc, fs=12.5)
arrow(ax, (45, 48), (45, 42), lw=1.6); label(ax, 47, 44.5, "orchestrator calls the capabilities", fs=12, ha="left", style="italic")
ax.add_patch(FancyBboxPatch((1, 11), 118, 30, boxstyle="round,pad=0,rounding_size=1", fc="none", ec="#888", lw=1.2, ls="--"))
label(ax, 60, 3, "Chapter 25 integrates, load-tests, and ships the whole system.", fs=12.5, style="italic")
save(fig, "fig_1_8_muse")
print("ok")
