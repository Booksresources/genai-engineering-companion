import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch16")
W = json.load(open("wf.json"))

fig, ax = canvas(13, 5.6, 124, 50)
lv = [("single call", "one prompt, one answer", C_GREY), ("workflow", "fixed steps defined in code", C_BLUE), ("agent", "model chooses tools and steps in a loop", C_AMBER), ("multi-agent", "several agents coordinate", C_ROSE)]
for i, (t, d, fc) in enumerate(lv):
    x = 2 + i * 30.5; box(ax, x, 22, 27, 11, t, fc, fs=14.5); label(ax, x + 13.5, 15, d, fs=13, width=18)
arrow(ax, (2, 6), (122, 6), lw=2.2); label(ax, 62, 2, "more autonomy, flexibility, cost, latency, and risk  →", fs=13.5, bold=True)
label(ax, 62, 45, "Choose the least autonomy that solves the task", fs=15, bold=True)
save(fig, "fig_16_1_autonomy")

fig, ax = canvas(13, 7.8, 124, 72)
ax.set_ylim(-4, 68)
label(ax, 2, 65, "Pipeline", fs=14.5, bold=True, ha="left")
for i, t in enumerate(["extract facts", "draft", "check", "format"]):
    box(ax, 2 + i * 30, 52, 25, 9, t, C_BLUE, fs=13.5)
    if i < 3: arrow(ax, (27.2 + i * 30, 56.5), (32 + i * 30, 56.5))
label(ax, 2, 44, "Orchestrator–workers", fs=14.5, bold=True, ha="left")
box(ax, 2, 26, 24, 12, "orchestrator: split task", C_AMBER, fs=13.5); box(ax, 92, 26, 30, 12, "orchestrator: combine results", C_AMBER, fs=13.5)
for i in range(3):
    box(ax, 44, 38 - i * 8, 30, 6.5, f"worker {i+1} (in parallel)", C_TEAL, fs=13); arrow(ax, (26.2, 32), (44, 41 - i * 8)); arrow(ax, (74.2, 41 - i * 8), (92, 32))
label(ax, 2, 20, "Generator–critic", fs=14.5, bold=True, ha="left")
box(ax, 2, 3, 30, 11, "generator drafts", C_LAV, fs=13.5); box(ax, 50, 3, 30, 11, "critic checks rules", C_ROSE, fs=13.5); box(ax, 96, 3, 26, 11, "accept", C_GREEN, fs=13.5)
arrow(ax, (32.2, 8.5), (50, 8.5)); arrow(ax, (80.2, 8.5), (96, 8.5)); arrow(ax, (65, 3), (17, 3), cs="arc3,rad=-0.25", ls="--"); label(ax, 41, -0.5, "revise (max 2)", fs=12.5)
save(fig, "fig_16_2_patterns")

fig, ax = canvas(13, 6.4, 124, 58)
label(ax, 62, 55, f"Research-and-draft workflow (real run: research {W['sequential']:.2f} s sequential, {W['parallel']:.2f} s parallel)", fs=14, bold=True)
st = [("plan: 4 questions", C_GREY, 1, 20), ("research ×4 in parallel (RAG)", C_TEAL, 25, 22), ("draft", C_LAV, 52, 18), ("critic: brand rules", C_ROSE, 75, 20), ("human approval", C_AMBER, 100, 22)]
for t, fc, x, w_ in st: box(ax, x, 26, w_, 14, t, fc, fs=13.5)
for a, b in [(21.2, 25), (47.2, 52), (70.2, 75), (95.2, 100)]: arrow(ax, (a, 33), (b, 33))
arrow(ax, (85, 26), (61, 26), cs="arc3,rad=-0.5", ls="--"); label(ax, 72, 13, f"draft 1 failed (exclamation mark, banned phrase, invented number);\nrevision 1 passed: \"{W['final_post']['body']}\"", fs=12.5)
save(fig, "fig_16_3_workflow")

fig, ax = canvas(13, 5.8, 124, 52)
label(ax, 62, 49, "Durable execution: resume from the last completed step", fs=15, bold=True)
steps = ["plan", "research ×4", "draft", "critic", "approval"]
for row, (lab, states) in enumerate([("first run", ["done", "done", "crash", "", ""]), ("resumed run", ["skipped", "skipped", "ran", "ran", "paused → approved"])]):
    y = 30 - row * 16; label(ax, 10, y + 4.5, lab, fs=14, bold=True)
    for i, (s, st_) in enumerate(zip(steps, states)):
        fc = {"done": C_TEAL, "crash": C_ROSE, "skipped": C_GREY, "ran": C_BLUE, "paused → approved": C_AMBER, "": "white"}[st_]
        box(ax, 22 + i * 20.3, y, 18.5, 9, f"{s}\n{st_}" if st_ else s, fc, fs=12.5)
save(fig, "fig_16_4_durable")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.6, 1]})
n = np.arange(1, 21)
for p, col in [(0.99, L_TEAL), (0.95, L_BLUE), (0.90, L_RED)]: a1.plot(n, 100 * p ** n, lw=2.6, color=col, label=f"{100*p:.0f}% per step")
a1.set_xlabel("Steps (calls or agents) in sequence"); a1.set_ylabel("Chance all succeed (%)"); a1.legend(frameon=False); a1.grid(True, color="#DDD"); a1.set_title("(a) Reliability compounds"); clean(a1)
v = [100 * W["retries_0"], 100 * W["retries_2"]]; a2.bar(["no retries", "2 retries"], v, color=[C_ROSE, C_TEAL], edgecolor=EDGE)
for i, x in enumerate(v): a2.text(i, x + 2, f"{x:.0f}%", ha="center", fontsize=14, fontweight="bold")
a2.set_ylabel("Workflows completed (%)"); a2.set_ylim(0, 110); a2.set_title("(b) 30% worker timeouts"); clean(a2)
fig.tight_layout(w_pad=3); save(fig, "fig_16_5_reliability")
print("ok")
