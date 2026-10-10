import sys, json, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
from vlm import *
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch13")
VL = json.load(open("vlm.json")); TO = json.load(open("text_only.json")); LR = json.load(open("lora_vlm.json"))

fig, ax = canvas(13, 6.0, 124, 54)
box(ax, 1, 30, 18, 14, "image (16×16 chart)", C_GREY, fs=13.5); box(ax, 23, 30, 20, 14, "vision encoder (CNN / ViT)", C_LAV, fs=13.5)
box(ax, 47, 30, 18, 14, "projector (MLP)", C_AMBER, fs=13.5); box(ax, 23, 8, 20, 12, "question tokens", C_GREY, fs=13.5)
box(ax, 72, 16, 26, 24, "language model (Transformer)", C_BLUE, fs=14); box(ax, 103, 22, 19, 12, "answer tokens", C_GREEN, fs=13.5)
arrow(ax, (19.2, 37), (23, 37)); arrow(ax, (43.2, 37), (47, 37)); arrow(ax, (65.2, 37), (72, 32)); arrow(ax, (43.2, 14), (72, 22)); arrow(ax, (98.2, 28), (103, 28))
label(ax, 56, 50, "16 visual tokens + question tokens in one sequence", fs=14, bold=True)
save(fig, "fig_13_1_architecture")

m = TinyVLM(); m.load_state_dict(torch.load("vlm.pt")); m.eval(); rng = np.random.default_rng(77)
fig = plt.figure(figsize=(13, 6.0))
for k, task in enumerate(["count", "tallest", "compare", "compare"]):
    im, q, a = example(rng, task)
    with torch.no_grad():
        qi = torch.tensor([[W[w] for w in q.split()] + [0] * (10 - len(q.split()))]); pred = WORDS[m(torch.tensor(im, dtype=torch.float32)[None, None], qi).argmax().item()]
    ax = fig.add_axes([0.01 + k * 0.13, 0.42, 0.11, 0.45]); ax.imshow(1 - (im + 1) / 2, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.065 + k * 0.13, 0.33, "\n".join(__import__("textwrap").wrap(q, 14)), ha="center", va="top", fontsize=11.5, color=BLACK)
    fig.text(0.065 + k * 0.13, 0.06, f"model: {pred}\ntrue: {a}", ha="center", fontsize=12.5, fontweight="bold", color=BLACK)
a2 = fig.add_axes([0.58, 0.14, 0.4, 0.74]); tasks = ["count", "tallest", "compare"]; x = np.arange(3); w = 0.36
a2.bar(x - w / 2, [100 * TO["acc"][t] for t in tasks], w, color=C_ROSE, edgecolor=EDGE, label="text only (no image)")
a2.bar(x + w / 2, [100 * VL["acc"][t] for t in tasks], w, color=C_BLUE, edgecolor=EDGE, label="vision-language model")
for i, t in enumerate(tasks):
    a2.text(i - w / 2, 100 * TO["acc"][t] + 2, f"{100*TO['acc'][t]:.0f}", ha="center", fontsize=12.5); a2.text(i + w / 2, 100 * VL["acc"][t] + 2, f"{100*VL['acc'][t]:.0f}", ha="center", fontsize=12.5)
a2.set_xticks(x); a2.set_xticklabels(["how many\nbars?", "which is\ntallest?", "is bar i taller\nthan bar j?"], fontsize=12.5)
a2.set_ylabel("Accuracy (%)"); a2.set_ylim(0, 118); a2.legend(frameon=False, loc="upper center", ncol=2, fontsize=11.5); clean(a2)
save(fig, "fig_13_2_chartqa")

fig, ax = canvas(13, 6.0, 124, 54)
label(ax, 62, 51, "Two ways to understand documents", fs=15, bold=True)
a_ = [("scanned page / PDF", C_GREY), ("OCR + layout analysis", C_AMBER), ("text with positions, tables", C_TEAL), ("LLM extracts fields", C_BLUE)]
for i, (t, fc) in enumerate(a_):
    x = 1 + i * 31; box(ax, x, 28, 27, 12, t, fc, fs=13.5)
    if i < 3: arrow(ax, (x + 27.2, 34), (x + 31, 34))
b_ = [("page image", C_GREY), ("vision-language model reads pixels directly", C_LAV), ("structured output (JSON schema)", C_GREEN)]
for i, (t, fc) in enumerate(b_):
    x = 1 + i * 42; box(ax, x, 6, 36, 12, t, fc, fs=13.5)
    if i < 2: arrow(ax, (x + 36.2, 12), (x + 42, 12))
label(ax, 2, 44, "OCR-based", fs=13.5, bold=True, ha="left"); label(ax, 2, 22, "OCR-free", fs=13.5, bold=True, ha="left")
save(fig, "fig_13_3_documents")

fig, ax = canvas(13, 5.8, 124, 52)
label(ax, 62, 49, "Voice assistants: cascade versus native audio model", fs=15, bold=True)
cc = [("speech in", C_GREY), ("ASR", C_AMBER), ("text LLM", C_BLUE), ("TTS", C_TEAL), ("speech out", C_GREY)]
for i, (t, fc) in enumerate(cc):
    x = 1 + i * 24.6; box(ax, x, 26, 21, 11, t, fc, fs=14)
    if i < 4: arrow(ax, (x + 21.2, 31.5), (x + 24.6, 31.5))
nn_ = [("speech in", C_GREY), ("audio-native multimodal model (audio tokens in and out)", C_LAV), ("speech out", C_GREY)]
for i, (t, fc) in enumerate(nn_):
    x = [1, 30, 103][i]; w_ = [21, 66, 21][i]; box(ax, x, 5, w_, 12, t, fc, fs=13.5)
arrow(ax, (22.2, 11), (30, 11)); arrow(ax, (96.2, 11), (103, 11))
save(fig, "fig_13_4_audio")

keys = ["before (zero-shot)", "LoRA on new task only", "full fine-tune on new task only", "full fine-tune, new + old mixed"]
labs = ["before", "LoRA,\nnew task only", "full FT,\nnew task only", "full FT,\nnew + old mixed"]
fig, ax = plt.subplots(figsize=(12, 5.6)); x = np.arange(4); w = 0.2
for j, (t, col, lab) in enumerate([("count", C_LAV, "count"), ("tallest", C_TEAL, "tallest"), ("compare", C_AMBER, "compare"), ("new", C_ROSE, "NEW: bars taller than bar 1")]):
    ax.bar(x + (j - 1.5) * w, [100 * LR[k][t] for k in keys], w, color=col, edgecolor=EDGE, label=lab)
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=13); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 110)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=12); clean(ax)
save(fig, "fig_13_5_lora")

fig, ax = canvas(13, 5.8, 124, 52)
st = [("upload: brief PDF, mood board, screenshots", C_GREY), ("VLM extracts structured brief (JSON schema)", C_BLUE), ("validate + ask user to confirm missing fields", C_AMBER),
      ("brief drives copy, images, voice, video", C_GREEN)]
for i, (t, fc) in enumerate(st):
    x = 1 + i * 31; box(ax, x, 18, 27, 18, t, fc, fs=13.5)
    if i < 3: arrow(ax, (x + 27.2, 27), (x + 31, 27))
label(ax, 62, 46, "Muse brief understanding", fs=15, bold=True)
label(ax, 62, 7, "fields: product, audience, key messages, tone, colors, must-avoid, deadline", fs=13, style="italic")
save(fig, "fig_13_6_muse")
print("ok")
