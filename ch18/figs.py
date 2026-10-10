import sys, json, torch
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch18")
ED = json.load(open("edit.json")); VD = json.load(open("voice_doc.json")); KT = json.load(open("kit.json")); EX = torch.load("edit_examples.pt")

fig = plt.figure(figsize=(13, 6.2))
ax = fig.add_axes([0, 0.55, 1, 0.43]); ax.set_xlim(0, 124); ax.set_ylim(0, 30); ax.axis("off")
st = [("\"Make the face smile\"", C_GREY), ("model → edit call {inpaint, mouth, smiling} (schema-validated)", C_BLUE), ("diffusion edit: inpaint or restyle", C_LAV), ("check: edit applied, rest unchanged", C_AMBER), ("show before/after + undo", C_GREEN)]
for i, (t, fc) in enumerate(st):
    x = 1 + i * 24.6; box(ax, x, 6, 21.5, 20, t, fc, fs=12.5)
    if i < 4: arrow(ax, (x + 21.7, 16), (x + 24.6, 16))
reqs = list(ED["results"].keys())
for k, req in enumerate(reqs):
    for j in range(2):
        a = fig.add_axes([0.02 + k * 0.155 + j * 0.07, 0.25, 0.065, 0.2]); im = np.array(EX[req][0][j])
        a.imshow(1 - (im + 1) / 2, cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([]); a.set_title("before" if j == 0 else "after", fontsize=11.5)
    fig.text(0.087 + k * 0.155, 0.17, ["smile", "add glasses", "grey background"][k], ha="center", fontsize=12.5, color=BLACK)
a2 = fig.add_axes([0.54, 0.1, 0.44, 0.4]); x = np.arange(3); w = 0.36
a2.bar(x - w / 2, [100 * ED["results"][r]["edit_succeeded"] for r in reqs], w, color=C_TEAL, edgecolor=EDGE, label="edit applied")
a2.bar(x + w / 2, [100 * ED["results"][r]["other_attributes_kept"] for r in reqs], w, color=C_BLUE, edgecolor=EDGE, label="everything else kept")
a2.set_xticks(x); a2.set_xticklabels(["smile", "glasses", "grey bg"], fontsize=12.5); a2.set_ylim(0, 115); a2.set_ylabel("% of 40"); a2.legend(frameon=False, ncol=2, loc="upper center", fontsize=11.5); clean(a2)
save(fig, "fig_18_1_editing")

V = VD["voice"]["components"]
fig, ax = plt.subplots(figsize=(12, 4.6))
seq = [("ASR", 0, V["asr_final"], C_AMBER), ("LLM, full reply", V["asr_final"], V["llm_ttft"] + 21 / V["llm_tokens_per_s"], C_BLUE), ("TTS", V["asr_final"] + V["llm_ttft"] + 21 / V["llm_tokens_per_s"], V["tts_first_audio"], C_TEAL)]
strm = [("ASR", 0, V["asr_final"], C_AMBER), ("LLM, 1st sentence", V["asr_final"], V["llm_ttft"] + 1 / V["llm_tokens_per_s"], C_BLUE), ("TTS", V["asr_final"] + V["llm_ttft"] + 1 / V["llm_tokens_per_s"], V["tts_first_audio"], C_TEAL)]
for row, items in [(1, seq), (0, strm)]:
    for lab, s, d, col in items:
        ax.barh(row, d, left=s, color=col, edgecolor=EDGE, height=0.5); ax.text(s + d / 2, row, lab, ha="center", va="center", fontsize=12, fontweight="bold")
ax.text(VD["voice"]["sequential_first_audio_s"] + 0.03, 1, f"{VD['voice']['sequential_first_audio_s']:.2f} s", va="center", fontsize=13.5, fontweight="bold")
ax.text(VD["voice"]["streamed_first_audio_s"] + 0.03, 0, f"{VD['voice']['streamed_first_audio_s']:.2f} s", va="center", fontsize=13.5, fontweight="bold")
ax.set_yticks([0, 1]); ax.set_yticklabels(["streamed by\nsentence", "wait for full\nreply"]); ax.set_xlabel("Seconds after the user stops speaking"); ax.set_xlim(0, 2.0); clean(ax)
save(fig, "fig_18_2_voice")

docs = VD["documents"]
fig = plt.figure(figsize=(13, 5.0))
ax = fig.add_axes([0, 0.05, 0.55, 0.9]); ax.set_xlim(0, 70); ax.set_ylim(0, 50); ax.axis("off")
st = [("document image / PDF", C_GREY, 40), ("extract fields (VLM or OCR + LLM)", C_BLUE, 29), ("validate: schema, totals, dates", C_AMBER, 18), ("pass → system; fail → human review", C_GREEN, 7)]
for t, fc, y in st: box(ax, 8, y, 54, 8, t, fc, fs=13.5)
for y in (40, 29, 18): arrow(ax, (35, y), (35, y - 2.8))
a2 = fig.add_axes([0.6, 0.2, 0.38, 0.7]); keys = ["clean", "digit", "date", "line"]
a2.bar(range(4), [100 * docs[k] for k in keys], color=[C_GREEN, C_ROSE, C_ROSE, C_ROSE], edgecolor=EDGE)
for i, k in enumerate(keys): a2.text(i, 100 * docs[k] + 2, f"{100*docs[k]:.0f}%", ha="center", fontsize=13.5, fontweight="bold")
a2.set_xticks(range(4)); a2.set_xticklabels(["clean", "misread\ntotal", "misread\ndate", "dropped\nline"], fontsize=12.5); a2.set_ylabel("Sent to human review (%)"); a2.set_ylim(0, 115); clean(a2)
save(fig, "fig_18_3_documents")

tm = KT["timings"]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 9.0))
names = [k.split(" (")[0][3:] for k in tm]; vals = list(tm.values())
a1.barh(range(len(vals))[::-1], vals, color=C_BLUE, edgecolor=EDGE)
for i, v in enumerate(vals): a1.text(v + 0.03, len(vals) - 1 - i, f"{v:.2f} s", va="center", fontsize=12.5)
a1.set_yticks(range(len(vals))[::-1]); a1.set_yticklabels(names, fontsize=12.5); a1.set_xlabel("Seconds (small models in this book, CPU)"); a1.set_xlim(0, 2.2)
a1.set_title(f"(a) Measured: {KT['sequential_s']:.2f} s sequential, {KT['critical_path_s']:.2f} s with parallel branches", fontsize=13.5); clean(a1)
prod = [("brief (VLM)", 0, 5, C_LAV), ("copy + check", 5, 4, C_BLUE), ("images ×4", 5, 15, C_TEAL), ("voiceover", 9, 3, C_AMBER), ("clip (async)", 20, 90, C_ROSE)]
for i, (lab, s, d, col) in enumerate(prod):
    a2.barh(len(prod) - 1 - i, d, left=s, color=col, edgecolor=EDGE); a2.text(s + d + 1, len(prod) - 1 - i, lab, va="center", fontsize=12.5)
a2.axvline(20, color=BLACK, ls="--", lw=1.3); a2.text(21, 4.45, "kit ready to review (~20 s)", fontsize=12)
a2.set_yticks([]); a2.set_xlabel("Seconds (illustrative production budget)"); a2.set_xlim(0, 160); a2.set_title("(b) Illustrative budget on hosted models", fontsize=13); clean(a2)
fig.tight_layout(h_pad=2.5); save(fig, "fig_18_4_budget")

fig, ax = canvas(13, 6.6, 124, 60)
box(ax, 1, 24, 20, 14, "one request: brief + product", C_GREY, fs=13)
box(ax, 26, 24, 18, 14, "brief extraction (Ch. 13)", C_LAV, fs=13)
br = [("copy + brand check (Ch. 5–8)", C_BLUE), ("images (Ch. 10)", C_TEAL), ("voiceover (Ch. 11)", C_AMBER), ("clip, async (Ch. 12)", C_ROSE)]
for i, (t, fc) in enumerate(br): box(ax, 52, 47 - i * 12, 30, 9, t, fc, fs=13); arrow(ax, (44.2, 31), (52, 51.5 - i * 12))
box(ax, 90, 24, 32, 14, "campaign kit for review: copy, 4 images, voice, clip, labels", C_GREEN, fs=13)
for i in range(4): arrow(ax, (82.2, 51.5 - i * 12), (90, 31))
arrow(ax, (21.2, 31), (26, 31)); label(ax, 62, 57.5, "Muse: from brief to campaign kit in one request", fs=15, bold=True)
label(ax, 62, 3, "branches run in parallel on the durable workflow engine (Ch. 16); every asset is checked, labeled, and logged", fs=12.5, style="italic")
save(fig, "fig_18_5_kit")
print("ok")
