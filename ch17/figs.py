import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch17")
ST = json.load(open("/home/claude/gb/ch17_stream.json"))["times"]

fig, ax = canvas(13, 7.4, 124, 68)
label(ax, 62, 65, "Five product patterns and what their interfaces need", fs=15, bold=True)
pats = [("Copilot", "suggest inside the user's tool; easy accept, edit, reject", C_BLUE), ("Extraction", "structured fields with source highlights; review queue", C_TEAL),
        ("Summarization", "length controls; links back to source passages", C_AMBER), ("Classification / routing", "confidence; human review below a threshold", C_LAV),
        ("Generation", "several candidates; regenerate; brand and safety checks; AI label", C_ROSE)]
for i, (t, d, fc) in enumerate(pats):
    y = 52 - i * 12; box(ax, 2, y, 32, 9.5, t, fc, fs=14.5); box(ax, 38, y, 84, 9.5, d, "white", fs=14, bold=False)
save(fig, "fig_17_1_patterns")

fig, a1 = plt.subplots(figsize=(11, 4.4))
a1.scatter(ST, np.ones(len(ST)), s=90, color=L_TEAL, edgecolor=BLACK, zorder=3, label="streamed words arrive")
a1.scatter([ST[-1]], [0], s=220, marker="D", color=L_RED, edgecolor=BLACK, zorder=3, label="non-streaming: everything at once")
a1.annotate(f"first words {ST[0]:.2f} s", (ST[0], 1), xytext=(0, 18), textcoords="offset points", ha="center", fontsize=13.5, fontweight="bold")
a1.annotate(f"{ST[-1]:.2f} s", (ST[-1], 0), xytext=(0, 16), textcoords="offset points", ha="center", fontsize=13.5, fontweight="bold")
a1.set_yticks([0, 1]); a1.set_yticklabels(["blocking", "streaming"]); a1.set_ylim(-0.7, 1.8); a1.set_xlim(0, 1.1); a1.set_xlabel("Seconds (real Muse server)")
a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=12.5); clean(a1)
save(fig, "fig_17_2_latency")

fig, ax = canvas(15.5, 9.0, 124, 70)
ax.add_patch(FancyBboxPatch((1, 1), 122, 68, boxstyle="round,pad=0,rounding_size=1", fc="#F4F1EA", ec=EDGE, lw=1.5))
ax.add_patch(FancyBboxPatch((1, 62), 122, 7, boxstyle="round,pad=0,rounding_size=1", fc="#2E5E4E", ec=EDGE, lw=1.5)); ax.text(4, 65.5, "Muse · Northwind marketing assistant", color="white", fontsize=15, fontweight="bold", va="center")
ax.add_patch(FancyBboxPatch((4, 4), 56, 55, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec="#9fb3aa")); label(ax, 6, 55.5, "Write a social post", fs=14.5, bold=True, ha="left")
for i, (k, v) in enumerate([("Product", "TrailMug"), ("Key feature", "keeps drinks hot for 6 hours")]):
    label(ax, 6, 50 - i * 7, k, fs=12.5, ha="left"); ax.add_patch(FancyBboxPatch((22, 47.5 - i * 7), 35, 5, boxstyle="round,pad=0,rounding_size=0.4", fc="white", ec="#9fb3aa")); label(ax, 23, 50 - i * 7, v, fs=12.5, ha="left")
box(ax, 6, 33, 12, 5, "Draft", "#D9732B", fs=13)
box(ax, 6, 19, 51, 12, "Meet the TrailMug. Keeps drinks hot for 6 hours. Built for long days outside. #Northwind", "white", fs=13, bold=False)
label(ax, 6, 16, "Draft 3f9a21c4 · prompt v2 · AI-generated: review first", fs=12, ha="left")
for i, t in enumerate(["Useful", "Not useful", "Copy", "Regenerate", "Undo"]): box(ax, 6 + i * 10.4, 7, 9.8, 5.5, t, "white", fs=11.5, bold=False)
ax.add_patch(FancyBboxPatch((64, 4), 56, 55, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec="#9fb3aa")); label(ax, 66, 55.5, "Ask the brand guidelines", fs=14.5, bold=True, ha="left")
ax.add_patch(FancyBboxPatch((66, 47.5), 51, 5, boxstyle="round,pad=0,rounding_size=0.4", fc="white", ec="#9fb3aa")); label(ax, 67, 50, "Which hex value is Fjord Green?", fs=12.5, ha="left")
box(ax, 66, 40, 10, 5, "Ask", "#D9732B", fs=13)
box(ax, 66, 26, 51, 11, "The primary color is Fjord Green, hex 2E5E4E.   [brand-2 · Color palette]", "white", fs=13, bold=False)
box(ax, 66, 8, 51, 12, "If not grounded: \"I couldn't find this in the guidelines.\"", "#FFF4E5", fs=12.5, bold=False)
save(fig, "fig_17_3_mockup")

fig, ax = canvas(13, 6.0, 124, 54)
box(ax, 1, 34, 30, 13, "explicit: thumbs, reasons, choices between drafts", C_BLUE, fs=13); box(ax, 1, 14, 30, 13, "implicit: copy, edit, regenerate, discard, publish", C_TEAL, fs=13)
box(ax, 40, 24, 24, 13, "feedback store (with prompt and model versions)", C_GREY, fs=13)
for i, t in enumerate(["dashboards: acceptance and edit rates", "new evaluation cases from failures", "preference pairs (Chapter 8)"]):
    box(ax, 74, 38 - i * 13, 48, 10, t, [C_AMBER, C_LAV, C_ROSE][i], fs=13.5); arrow(ax, (64.2, 30.5), (74, 43 - i * 13))
arrow(ax, (31.2, 40), (40, 33)); arrow(ax, (31.2, 21), (40, 28))
label(ax, 62, 51, "Feedback capture and where it goes", fs=15, bold=True)
label(ax, 62, 3, "edit similarity of a published draft: 1.0 = used as is; our test edit scored between 0.5 and 1.0", fs=12.5, style="italic")
save(fig, "fig_17_4_feedback")

fig, ax = canvas(13, 5.6, 124, 50)
box(ax, 1, 16, 26, 18, "browser: index.html + EventSource + fetch", C_GREY, fs=13.5)
eps = [("GET /  → page", C_LAV), ("GET /api/draft/stream  → server-sent events", C_TEAL), ("GET /api/ask  → cited answer (Ch. 14)", C_BLUE), ("POST /api/feedback  → JSONL store", C_AMBER)]
for i, (t, fc) in enumerate(eps):
    box(ax, 40, 38 - i * 10.5, 52, 8, t, fc, fs=13); arrow(ax, (27.2, 25), (40, 42 - i * 10.5))
box(ax, 100, 16, 22, 18, "FastAPI app (muse.web.app), tests: 4", C_GREEN, fs=13)
label(ax, 62, 47.5, "Muse web front end", fs=15, bold=True)
save(fig, "fig_17_5_frontend")
print("ok")
