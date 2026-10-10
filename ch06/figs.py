import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch06")
R = json.load(open("prompt_eval.json"))

# 6.1 anatomy
fig, ax = canvas(12, 7.6, 120, 70)
parts = [("System: role and audience", "You are Muse, Northwind Studio's copywriter.", C_LAV),
         ("Instructions and rules", "Warm, concise; no exclamation marks; never invent specifications.", C_BLUE),
         ("Output format", "JSON with headline (≤ 60 chars), body (≤ 280), hashtags (1–3).", C_TEAL),
         ("Examples (few-shot)", "One or more input → ideal output pairs.", C_AMBER),
         ("Context", "Product facts, retrieved brand guidelines, conversation history.", C_GREEN),
         ("Task input", "Product: TrailMug. Features: keeps drinks hot 6 hours.", C_ROSE)]
for i, (t, ex, fc) in enumerate(parts):
    y = 58 - i * 11
    box(ax, 2, y, 34, 9, t, fc, fs=14.5)
    box(ax, 40, y, 78, 9, ex, "white", fs=14, bold=False)
label(ax, 60, 68, "Anatomy of a production prompt (stable parts first, variable parts last)", fs=15, bold=True)
save(fig, "fig_6_1_anatomy")

# 6.2 patterns
fig, ax = canvas(12, 9.0, 120, 82)
pats = [("Zero-shot", "Instruction only. Start here.", C_GREY), ("Few-shot", "Add 2–5 worked examples that show the format and judgment you want.", C_AMBER),
        ("Chain-of-thought", "Ask the model to reason step by step before answering; keep reasoning out of the final output.", C_BLUE),
        ("Decomposition", "Split the task into steps or calls: extract facts → draft → check.", C_TEAL),
        ("Self-critique", "Generate, then critique against rules, then revise.", C_LAV)]
for i, (t, d, fc) in enumerate(pats):
    y = 64 - i * 15.5
    box(ax, 2, y, 30, 13, t, fc, fs=15); box(ax, 36, y, 82, 13, d, "white", fs=14, bold=False)
label(ax, 60, 80, "Five prompting patterns, from simplest to most elaborate", fs=15, bold=True)
save(fig, "fig_6_2_patterns")

# 6.3 versioning lifecycle
fig, ax = canvas(12, 5.6, 120, 50)
st = [("Edit prompt in Git", C_GREY), ("New version and fingerprint", C_LAV), ("Run eval suite (dev and test)", C_BLUE), ("Compare with production version", C_AMBER), ("Promote, or keep current", C_GREEN)]
for i, (t, fc) in enumerate(st):
    x = 1 + i * 24; box(ax, x, 22, 21.5, 16, t, fc, fs=14)
    if i < 4: arrow(ax, (x + 21.7, 30), (x + 24, 30))
arrow(ax, (108, 22), (12, 22), cs="arc3,rad=-0.3", ls="--"); label(ax, 60, 4, "every response logs prompt name, version, and fingerprint", fs=13.5, style="italic")
save(fig, "fig_6_3_versioning")

# 6.4 hand-written variants (real)
H = R["hand"]; names = list(H.keys()); tasks = ["reverse", "upper", "sort", "add"]
fig, ax = plt.subplots(figsize=(12, 5.8))
x = np.arange(len(tasks) + 1); w = 0.27; cols = [C_ROSE, C_AMBER, C_TEAL]
labels = {"v1-natural": "\"Please reverse this: …\"", "v2-verb-colon": "\"Reverse: …\"", "v3-training-format": "\"reverse: …\""}
for k, (n, col) in enumerate(zip(names, cols)):
    vals = [100 * H[n]["by_task"][t] for t in tasks] + [100 * H[n]["overall"]]
    ax.bar(x + (k - 1) * w, vals, w, color=col, edgecolor=EDGE, label=labels[n])
    for xi, v in zip(x + (k - 1) * w, vals): ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=12.5)
ax.set_xticks(x); ax.set_xticklabels(tasks + ["overall"]); ax.set_ylabel("Exact-match accuracy (%)"); ax.set_ylim(0, 112)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3); clean(ax)
save(fig, "fig_6_4_variants")

# 6.5 search distribution (real)
tr = R["search"]["trace"]
fig, ax = plt.subplots(figsize=(12, 5.6))
rng = np.random.default_rng(0)
for i, t in enumerate(tr):
    a = np.array(t["all"]) * 100
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(a)), a, s=36, color=L_BLUE, alpha=0.6, edgecolor="none")
    ax.scatter(i, t["best"][0] * 100, s=320, marker="*", color=L_RED, edgecolor=BLACK, zorder=5)
    ax.annotate(f"\"{t['best'][1].replace('{x}', '…')}\"", (i, t["best"][0] * 100), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=13, fontweight="bold")
ax.set_xticks(range(len(tr))); ax.set_xticklabels([t["task"] for t in tr]); ax.set_ylabel("Dev-set accuracy (%)"); ax.set_ylim(-5, 112)
ax.set_title("64 candidate templates per task (dots); best found (star)"); ax.grid(True, axis="y", color="#DDD"); clean(ax)
save(fig, "fig_6_5_search")

# 6.6 eval harness
fig, ax = canvas(12, 6.0, 120, 54)
box(ax, 1, 34, 22, 14, "Test cases: inputs + expected outputs or rules", C_GREY, fs=13.5)
box(ax, 1, 12, 22, 14, "Prompt version (template + decoding)", C_LAV, fs=13.5)
box(ax, 30, 22, 20, 14, "Render prompts", C_BLUE, fs=14); box(ax, 57, 22, 18, 14, "Model", C_AMBER, fs=14)
box(ax, 82, 34, 36, 14, "Scorers: exact match, rule checks, schema, LLM judge", C_TEAL, fs=13.5)
box(ax, 82, 10, 36, 14, "Report by slice; gate: block if worse than production", C_GREEN, fs=13.5)
arrow(ax, (23.2, 41), (30, 31)); arrow(ax, (23.2, 19), (30, 27)); arrow(ax, (50.2, 29), (57, 29)); arrow(ax, (75.2, 31), (82, 40)); arrow(ax, (100, 34), (100, 24.2))
save(fig, "fig_6_6_harness")
print("ok")
