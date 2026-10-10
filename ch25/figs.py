import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch25")
LA = json.load(open("launch.json")); G = json.load(open("/home/claude/gb/ch19/gate.json")); LD = json.load(open("/home/claude/gb/ch23/load.json"))

fig, ax = canvas(13, 8.4, 124, 78)
label(ax, 62, 75, "Muse: final architecture", fs=15, bold=True)
box(ax, 2, 60, 26, 9, "web front end (Ch. 17)", C_GREY, fs=13); box(ax, 32, 60, 26, 9, "API + workflows (Ch. 5, 16)", C_BLUE, fs=13)
box(ax, 62, 60, 28, 9, "safety layer (Ch. 20)", C_ROSE, fs=13); box(ax, 94, 60, 28, 9, "gateway: routing, fallback, cache (Ch. 23–24)", C_LAV, fs=12)
caps = [("copy: prompts + LoRA (Ch. 6–8)", C_BLUE), ("brand Q&A: RAG (Ch. 14)", C_TEAL), ("agent + MCP tools (Ch. 15)", C_AMBER), ("images (Ch. 9–10)", C_AMBER),
        ("voice (Ch. 11)", C_TEAL), ("clips, async (Ch. 12)", C_ROSE), ("briefs: VLM (Ch. 13)", C_LAV), ("campaign kit (Ch. 18)", C_GREEN)]
for i, (t, fc) in enumerate(caps):
    x = 2 + (i % 4) * 30.5; y = 40 - (i // 4) * 12; box(ax, x, y, 28, 9, t, fc, fs=12.5)
ax.add_patch(FancyBboxPatch((1, 27), 122, 24, boxstyle="round,pad=0,rounding_size=1", fc="none", ec="#888", lw=1.2, ls="--"))
ops = [("release gate (Ch. 19)", C_GREEN), ("governance + system card (Ch. 21)", C_GREY), ("serving + autoscaling (Ch. 22–23)", C_BLUE), ("tracing, monitors, flags (Ch. 24)", C_TEAL)]
for i, (t, fc) in enumerate(ops): box(ax, 2 + i * 30.5, 4, 28, 12, t, fc, fs=12.5)
label(ax, 62, 21, "operated and governed by", fs=13, style="italic")
for a, b in [(28.2, 32), (58.2, 62), (90.2, 94)]: arrow(ax, (a, 64.5), (b, 64.5))
save(fig, "fig_25_1_architecture")

fig, ax = canvas(13, 6.0, 124, 54)
label(ax, 62, 51, "Pre-launch checks (all real results from this book)", fs=15, bold=True)
checks = [("automated tests", LA["tests"].split(" in ")[0], True), ("release gate", "PASS (5 metrics)" if G["base"]["passed"] else "BLOCKED", G["base"]["passed"]),
          ("load test p95 at 16 clients", f"{LD['load']['16']['p95_ms']:.0f} ms", True), ("safety tests", "injection blocked; tampering detected", True),
          ("system card", "generated from gate results", True), ("rollback", "release fingerprint + kill switches", True)]
for i, (k, v, ok) in enumerate(checks):
    x = 2 + (i % 3) * 40.5; y = 26 - (i // 3) * 20
    box(ax, x, y + 8, 38, 8, k, C_GREY, fs=13.5); box(ax, x, y, 38, 8, v, C_GREEN if ok else C_ROSE, fs=13, bold=False)
save(fig, "fig_25_2_checks")

tl = LA["rollout_v3"]["timeline"]
fig, ax = plt.subplots(figsize=(12, 5.4))
hrs = [t["hour"] + 1 for t in tl]; ax.plot(hrs, [100 * t["new_rate"] for t in tl], "-o", ms=8, lw=2.6, color=L_RED, label="v3 (candidate) violation rate")
ax.plot(hrs, [100 * t["old_rate"] for t in tl], "-s", ms=8, lw=2.6, color=L_TEAL, label="v2 (current) violation rate")
for t in tl: ax.text(t["hour"] + 1, 100 * t["new_rate"] + 0.7, t["stage"], ha="center", fontsize=11.5)
ax.axvline(len(tl), color=BLACK, ls="--", lw=1.4); ax.text(len(tl) - 0.1, 11.5, f"automatic rollback\nafter {len(tl)} h at the {LA['rollout_v3']['stage']} stage", ha="right", fontsize=12.5, fontweight="bold")
ax.set_xlabel("Hours since rollout began"); ax.set_ylabel("Flagged posts (%)"); ax.set_ylim(0, 14); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2); clean(ax)
save(fig, "fig_25_3_rollout")

fig, ax = canvas(13, 7.6, 124, 70)
label(ax, 62, 67, "Ten lessons from building Muse", fs=15, bold=True)
lessons = ["Start with the strongest model, a clear prompt, and an eval set", "Measure everything on slices, with intervals", "Prefer code checks; use judges with care",
           "Retrieve knowledge; fine-tune behavior", "Workflows before agents; least privilege always", "Treat every input as untrusted", "Label and sign what you generate",
           "Optimize in a measured loop; revert what does not help", "Version everything; gate every release", "Roll out in stages with kill switches"]
for i, t in enumerate(lessons):
    x = 2 + (i % 2) * 61; y = 52 - (i // 2) * 12; box(ax, x, y, 58, 9.5, f"{i+1}. {t}", [C_BLUE, C_TEAL, C_AMBER, C_LAV, C_ROSE][i // 2], fs=13)
save(fig, "fig_25_4_lessons")
print("ok")
