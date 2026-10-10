import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch04")
from matplotlib.ticker import FuncFormatter
TK = json.load(open("tok.json")); AT = json.load(open("attn.json")); TR = json.load(open("train_tiny.json")); DE = json.load(open("decode.json"))

# 4.1 tokenization pipeline with real splits
fig, ax = canvas(13, 6.6, 120, 60)
label(ax, 60, 57, "From text to token IDs (16,000-entry BPE trained on Shakespeare)", fs=15, bold=True)
rows = TK["splits"]["16000"]
y = 46
for r in rows:
    label(ax, 1, y + 3, f"\"{r['text']}\"", fs=13, ha="left", bold=True)
    x = 1
    for t in r["tokens"]:
        disp = t.replace(" ", "␣")
        w = max(3.2, 1.25 * len(disp) + 1.6)
        if x + w > 118: break
        ax.add_patch(FancyBboxPatch((x, y - 4.2), w, 5, boxstyle="round,pad=0,rounding_size=0.4", fc=C_BLUE, ec=EDGE, lw=1))
        ax.text(x + w / 2, y - 1.7, disp, ha="center", va="center", fontsize=12.5, family="DejaVu Sans Mono", color=BLACK)
        x += w + 0.6
    label(ax, 119, y + 3, f"{len(r['tokens'])} tokens", fs=12.5, ha="right")
    y -= 12.5
label(ax, 60, 1.5, "␣ marks a leading space. Common words are single tokens; rare words, names, and numbers split into pieces.", fs=12.5, style="italic")
save(fig, "fig_4_1_tokens")

# 4.2 compression vs vocab
cp = TK["compression"]; Vs = list(cp.keys())
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.bar(range(len(Vs)), [cp[v]["chars_per_token"] for v in Vs], color=[C_GREY, C_BLUE, C_TEAL, C_AMBER], edgecolor=EDGE, width=0.6)
for i, v in enumerate(Vs): ax.text(i, cp[v]["chars_per_token"] + 0.06, f"{cp[v]['chars_per_token']:.2f}", ha="center", fontsize=14, fontweight="bold")
ax.set_xticks(range(len(Vs))); ax.set_xticklabels([f"{int(v):,}" for v in Vs]); ax.set_xlabel("Vocabulary size"); ax.set_ylabel("Characters per token (held-out text)")
ax.set_ylim(0, 3.9); clean(ax); ax.set_title("Bigger vocabularies compress text into fewer tokens")
save(fig, "fig_4_2_compression")

# 4.3 attention by hand
man = AT["manual"]; Wm = np.array(man["weights"]); toks = man["tokens"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})
a1.axis("off"); a1.set_xlim(0, 60); a1.set_ylim(0, 50)
st = [("1. scores = Q Kᵀ / √d", C_BLUE), ("2. mask future positions (−∞)", C_ROSE), ("3. softmax each row → weights", C_AMBER), ("4. output = weights × V", C_TEAL)]
for i, (t, fc) in enumerate(st):
    box(a1, 2, 38 - i * 11.5, 56, 8.5, t, fc, fs=14)
    if i < 3: arrow(a1, (30, 38 - i * 11.5), (30, 38 - (i + 1) * 11.5 + 8.7))
im = a2.imshow(Wm, cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        a2.text(j, i, "—" if j > i else f"{Wm[i, j]:.2f}", ha="center", va="center", fontsize=15, fontweight="bold", color=BLACK)
a2.set_xticks(range(4)); a2.set_xticklabels(toks); a2.set_yticks(range(4)); a2.set_yticklabels(toks)
a2.set_xlabel("attends to (key)"); a2.set_ylabel("token (query)"); a2.set_title("Attention weights (each row sums to 1)")
fig.tight_layout(w_pad=2); save(fig, "fig_4_3_attention_hand")

# 4.4 real attention maps
pr = AT["prompt"]; labs = [ch if ch not in "\n " else ("⏎" if ch == "\n" else "␣") for ch in pr]
fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
for li, ax in enumerate(axes):
    M = np.array(AT["maps"][li][0]); ax.imshow(M, cmap="Blues", vmin=0, vmax=min(1, M.max()))
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=11, family="DejaVu Sans Mono")
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=11, family="DejaVu Sans Mono")
    ax.set_title(f"layer {li+1}, head 1", fontsize=14)
fig.suptitle("Attention in the trained tiny GPT for \"ROMEO:⏎But soft, what light\" (rows: query position)", fontsize=14, fontweight="bold", color=BLACK)
fig.tight_layout(); save(fig, "fig_4_4_attention_real")

# 4.5 architecture
fig, ax = canvas(13, 7.2, 120, 66)
box(ax, 2, 4, 22, 8, "token IDs", C_GREY, fs=13.5); box(ax, 2, 16, 22, 8, "token embedding", C_LAV, fs=13.5)
arrow(ax, (13, 12.2), (13, 16))
ax.add_patch(FancyBboxPatch((30, 6), 52, 52, boxstyle="round,pad=0,rounding_size=1", fc="#F7F9FC", ec=EDGE, lw=1.5, ls="--"))
label(ax, 56, 61.5, "Transformer block × 4", fs=14.5, bold=True)
blk = [("RMSNorm", C_GREY), ("masked multi-head attention (with RoPE positions)", C_BLUE), ("add residual", C_GREY), ("RMSNorm", C_GREY), ("SwiGLU feed-forward network", C_AMBER), ("add residual", C_GREY)]
for i, (t, fc) in enumerate(blk):
    box(ax, 34, 9 + i * 8, 44, 6.4, t, fc, fs=13)
    if i < 5: arrow(ax, (56, 15.6 + i * 8), (56, 17 + i * 8), ms=10)
arrow(ax, (24.2, 20), (34, 12.2))
box(ax, 90, 46, 26, 8, "final RMSNorm", C_GREY, fs=13.5); box(ax, 90, 32, 26, 9, "linear head → logits over vocabulary", C_TEAL, fs=13)
box(ax, 90, 17, 26, 9, "softmax → next-token probabilities", C_GREEN, fs=13)
arrow(ax, (78.2, 52), (90, 50)); arrow(ax, (103, 46), (103, 41.2)); arrow(ax, (103, 32), (103, 26.2))
save(fig, "fig_4_5_architecture")

# 4.6 training curve + samples
lg = np.array(TR["log"])
fig = plt.figure(figsize=(13, 6.6))
a1 = fig.add_axes([0.07, 0.15, 0.36, 0.75])
a1.plot(lg[:, 0], lg[:, 1], "-o", lw=2.4, color=L_BLUE, label="training loss"); a1.plot(lg[:, 0], lg[:, 2], "-s", lw=2.4, color=L_RED, label="validation loss")
a1.axhline(np.log(65), color="#777", ls=":", lw=1.4); a1.text(500, np.log(65) - 0.12, "random guessing (4.17)", ha="right", fontsize=12.5)
a1.set_xlabel("Training step"); a1.set_ylabel("Cross-entropy (nats per character)"); a1.legend(frameon=False); a1.grid(True, color="#DDD"); clean(a1); a1.set_ylim(1.3, 4.4)
a2 = fig.add_axes([0.48, 0.02, 0.51, 0.96]); a2.axis("off"); a2.set_xlim(0, 60); a2.set_ylim(0, 60)
for i, (k, v) in enumerate(TR["samples"].items()):
    y = 50 - i * 11.5
    box(a2, 0.5, y, 10, 9, f"step {k}", C_GREY, fs=13)
    a2.add_patch(FancyBboxPatch((11.5, y), 48, 9, boxstyle="round,pad=0,rounding_size=0.5", fc="white", ec=EDGE, lw=1))
    t, f = wrap_text(a2, v.replace("\n", " ⏎ ")[:80], 46, 9, 12)
    a2.text(12.5, y + 4.5, t, va="center", fontsize=f, family="DejaVu Sans Mono", color=BLACK)
save(fig, "fig_4_6_training")

# 4.7 scale
items = [("Our tiny GPT", 7.5e5, 2.05e6), ("Typical 7B open model", 7e9, 2e12), ("Frontier-scale dense model (≈400B)", 4e11, 1.5e13)]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
for ax, (title, idx) in zip(axes, [("Parameters", 1), ("Training tokens", 2), ("Training compute (FLOPs ≈ 6ND)", 3)]):
    vals = [it[1] if idx == 1 else it[2] if idx == 2 else 6 * it[1] * it[2] for it in items]
    ax.bar(range(3), vals, color=[C_TEAL, C_BLUE, C_ROSE], edgecolor=EDGE, width=0.6); ax.set_yscale("log")
    for i, v in enumerate(vals): ax.text(i, v * 2.2, f"{v:.1e}", ha="center", fontsize=12.5, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(["tiny GPT", "7B model", "≈400B model"], fontsize=13); ax.set_title(title)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"1e{int(np.log10(v))}" if v > 0 else "")); ax.set_ylim(min(vals) / 5, max(vals) * 40); clean(ax)
fig.tight_layout(w_pad=2); save(fig, "fig_4_7_scale")

# 4.8 decoding
fig = plt.figure(figsize=(13, 6.0))
ax = fig.add_axes([0.07, 0.12, 0.5, 0.82])
cols = [L_RED, L_PURPLE, L_TEAL, L_BLUE, "#555555", L_AMBER, L_GREEN, "#8B5A2B", "#1A1A6E"]
for i, ((k, v), col) in enumerate(zip(DE.items(), cols)):
    ax.scatter(100 * v["distinct8"], 100 * v["real_words"], s=330, color=col, edgecolor=BLACK, zorder=5)
    ax.text(100 * v["distinct8"], 100 * v["real_words"], str(i + 1), ha="center", va="center", fontsize=12, fontweight="bold", color="white", zorder=6)
ax.set_xlabel("Diversity: distinct 8-character sequences (%)"); ax.set_ylabel("Quality: real words (%)")
ax.set_xlim(55, 103); ax.set_ylim(70, 101); ax.grid(True, color="#DDD"); clean(ax)
ax2 = fig.add_axes([0.6, 0.05, 0.39, 0.9]); ax2.axis("off"); ax2.set_xlim(0, 40); ax2.set_ylim(0, 60)
for i, ((k, v), col) in enumerate(zip(DE.items(), cols)):
    y = 55 - i * 6.2
    ax2.scatter(1.5, y, s=330, color=col, edgecolor=BLACK); ax2.text(1.5, y, str(i + 1), ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax2.text(3.5, y, f"{k}: {100*v['real_words']:.0f}% real, {100*v['distinct8']:.0f}% distinct", va="center", fontsize=13, color=BLACK)
save(fig, "fig_4_8_decoding")
print("ok")
