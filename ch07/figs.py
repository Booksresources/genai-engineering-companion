import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch07")
D = json.load(open("brand_data.json")); R = json.load(open("ft_results.json")); MG = json.load(open("merge.json"))

# 7.1 decision
fig, ax = canvas(12, 8.4, 120, 76)
qs = [("Does a strong model with a good prompt and examples already pass your evaluation?", "Ship the prompt (Chapter 6)"),
      ("Is the gap missing knowledge or facts?", "Use retrieval (Chapter 14)"),
      ("Is the gap format, style, a narrow skill, or cost at scale?", "Fine-tune with LoRA or QLoRA"),
      ("Do you have hundreds to thousands of clean, representative examples?", "If not: collect data first")]
for i, (q, a) in enumerate(qs):
    y = 58 - i * 18
    box(ax, 2, y, 66, 14, q, C_AMBER, fs=14); box(ax, 80, y + 2, 38, 10, a, C_TEAL, fs=14)
    arrow(ax, (68.2, y + 7), (80, y + 7)); label(ax, 74, y + 9.5, "yes" if i < 3 else "no", fs=13, bold=True)
    if i < 3: arrow(ax, (35, y), (35, y - 3.8)); label(ax, 37.5, y - 2, "no" if i < 2 else "", fs=13, bold=True, ha="left")
save(fig, "fig_7_1_when")

# 7.2 cleaning funnel (real)
rep = D["report"]; steps = [("raw examples", rep["raw"])] + [(k, -v) for k, v in rep.items() if k not in ("raw", "kept")] + [("clean examples kept", rep["kept"])]
fig, ax = plt.subplots(figsize=(12, 5.8))
names = [s[0] for s in steps]; vals = [abs(s[1]) for s in steps]
cols = [C_GREY] + [C_ROSE] * (len(steps) - 2) + [C_TEAL]
ax.barh(range(len(steps))[::-1], vals, color=cols, edgecolor=EDGE)
for i, (n, v) in enumerate(steps):
    ax.text(abs(v) + 12, len(steps) - 1 - i, (f"−{abs(v)}" if v < 0 else f"{v}"), va="center", fontsize=14, fontweight="bold")
ax.set_yticks(range(len(steps))[::-1]); ax.set_yticklabels(names, fontsize=14); ax.set_xlabel("Examples"); ax.set_xlim(0, 1450); clean(ax)
save(fig, "fig_7_2_cleaning")

# 7.3 LoRA
fig, ax = canvas(12.5, 5.8, 124, 52)
box(ax, 1, 20, 14, 12, "input x", C_GREY, fs=14.5)
box(ax, 22, 30, 32, 16, "frozen base weight W", C_BLUE, fs=15)
box(ax, 22, 4, 14, 16, "A (r × d)", C_TEAL, fs=14); box(ax, 40, 4, 14, 16, "B (d × r), starts at 0", C_TEAL, fs=13.5)
arrow(ax, (15.2, 28), (22, 38)); arrow(ax, (15.2, 24), (22, 12)); arrow(ax, (36.2, 12), (40, 12))
ax.scatter([66], [26], s=700, color="white", edgecolor=BLACK, lw=1.8, zorder=4); ax.text(66, 26, "+", ha="center", va="center", fontsize=22, fontweight="bold", zorder=5)
arrow(ax, (54.2, 38), (64.6, 27.5)); arrow(ax, (54.2, 12), (64.6, 24.5)); label(ax, 60, 14.5, "× α/r", fs=14)
box(ax, 74, 20, 14, 12, "output", C_GREY, fs=14.5); arrow(ax, (67.6, 26), (74, 26))
label(ax, 108, 32, "Only A and B train:\nr(d_in + d_out)\nper layer", fs=14, bold=True)
label(ax, 108, 16, "Our model:\n74,752 trainable\nof 746,752", fs=14)
save(fig, "fig_7_3_lora")

# 7.4 training curves
fig, ax = plt.subplots(figsize=(11, 5.6))
for k, col, lab in [("lora", L_BLUE, "LoRA (r = 8)"), ("qlora", L_AMBER, "QLoRA (4-bit NF4 base)"), ("full", L_RED, "full fine-tuning")]:
    h = np.array(R[k]["hist"]); ax.plot(h[:, 0], h[:, 1], "-o", lw=2.6, ms=6, color=col, label=lab)
ax.set_xlabel("Training step"); ax.set_ylabel("Loss on copy (nats/char)"); ax.legend(frameon=False); ax.grid(True, color="#DDD"); clean(ax)
save(fig, "fig_7_4_curves")

# 7.5 results
modes = [("prompt_base", "prompting the\nbase model"), ("lora", "LoRA"), ("qlora", "QLoRA"), ("full", "full\nfine-tuning")]
fig, ax = plt.subplots(figsize=(12, 6.0))
x = np.arange(len(modes)); w = 0.2
metrics = [("pass_rate", "passes all brand rules", C_GREEN), ("name", "correct product name", C_BLUE), ("feature", "feature copied exactly", C_AMBER), ("signoff", "ends with \"Northwind.\"", C_LAV)]
for j, (k, lab, col) in enumerate(metrics):
    vals = [100 * (R[m]["pass_rate"] if k == "pass_rate" else R[m]["parts"][k]) for m, _ in modes]
    ax.bar(x + (j - 1.5) * w, vals, w, color=col, edgecolor=EDGE, label=lab)
    for xi, v in zip(x + (j - 1.5) * w, vals): ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=12)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in modes]); ax.set_ylabel("Share of 60 unseen products (%)"); ax.set_ylim(0, 115)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2); clean(ax)
save(fig, "fig_7_5_results")

# 7.6 forgetting
fig, ax = plt.subplots(figsize=(10, 5.2))
vals = [R[m]["shakespeare_loss"] for m, _ in modes]
ax.bar(range(4), vals, color=[C_GREY, C_BLUE, C_AMBER, C_ROSE], edgecolor=EDGE, width=0.6)
for i, v in enumerate(vals): ax.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=14, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(["base model", "after LoRA", "after QLoRA", "after full FT"]); ax.set_ylabel("Shakespeare loss"); ax.set_ylim(0, 4)
ax.set_title("Forgetting: the old skill degrades after narrow fine-tuning"); clean(ax)
save(fig, "fig_7_6_forgetting")

# 7.7 serving adapters
fig, ax = canvas(12, 6.0, 120, 54)
label(ax, 30, 51, "Option A: merge", fs=15, bold=True); label(ax, 90, 51, "Option B: keep adapters separate", fs=15, bold=True)
box(ax, 4, 30, 22, 12, "base weights", C_BLUE, fs=14); box(ax, 30, 30, 22, 12, f"adapter ({MG['adapter_kb']:.0f} KB)", C_TEAL, fs=14)
arrow(ax, (28, 30), (28, 24)); box(ax, 10, 8, 36, 14, f"one merged model ({MG['merged_kb']/1024:.1f} MB); identical outputs, no extra latency", C_GREEN, fs=13.5)
box(ax, 66, 30, 22, 14, "one shared base model", C_BLUE, fs=14)
for i, t in enumerate(["brand-voice adapter", "support adapter", "customer X adapter"]):
    box(ax, 94, 38 - i * 11, 24, 8.5, t, C_TEAL, fs=13)
    arrow(ax, (88.2, 37), (94, 42 - i * 11))
label(ax, 90, 4, "swap or batch adapters per request (Chapter 23)", fs=13.5, style="italic")
save(fig, "fig_7_7_serving")
print("ok")
