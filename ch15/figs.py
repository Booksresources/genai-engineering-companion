import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch15")
AG = json.load(open("agent.json")); MS = json.load(open("memory_sandbox.json"))

fig, ax = canvas(13, 5.8, 124, 52)
st = [("app sends messages + tool schemas", C_GREY), ("model returns a tool call (name + JSON arguments)", C_BLUE), ("app validates, checks permissions, executes", C_AMBER),
      ("app returns the result as a tool message", C_TEAL), ("model answers or calls another tool", C_GREEN)]
for i, (t, fc) in enumerate(st):
    x = 1 + i * 24.6; box(ax, x, 18, 21.5, 18, t, fc, fs=13)
    if i < 4: arrow(ax, (x + 21.7, 27), (x + 24.6, 27))
arrow(ax, (110, 18), (60, 10), cs="arc3,rad=-0.2", ls="--"); label(ax, 88, 7, "loop", fs=13, style="italic")
label(ax, 62, 47, "Function calling: the model proposes, the application disposes", fs=15, bold=True)
save(fig, "fig_15_1_function_calling")

fig, ax = canvas(13, 6.6, 124, 60)
label(ax, 62, 57, "A real run of the Muse scheduling agent (4 tool calls)", fs=15, bold=True)
for i, t in enumerate(AG["trace"]):
    y = 43 - i * 11; box(ax, 1, y, 28, 8.5, t["tool"], [C_TEAL, C_AMBER, C_LAV, C_ROSE][i], fs=12.5)
    box(ax, 31, y, 91, 8.5, t["result"].replace('"', "")[:110], "white", fs=12.5, bold=False)
label(ax, 62, 2.5, "final: " + AG["final"].replace("'", ""), fs=13.5, bold=True)
save(fig, "fig_15_2_trace")

g = AG["reliability"]; fails = [0.0, 0.05, 0.1, 0.2]
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.plot([100 * f for f in fails], [100 * g[f"naive|{f}"] for f in fails], "-o", ms=9, lw=2.6, color=L_RED, label="naive loop (no validation, no retries)")
ax.plot([100 * f for f in fails], [100 * g[f"robust|{f}"] for f in fails], "-s", ms=9, lw=2.6, color=L_TEAL, label="robust loop (validate, retry ×2, step budget)")
for f in fails: ax.annotate(f"{100*g[f'naive|{f}']:.0f}%", (100 * f, 100 * g[f"naive|{f}"]), xytext=(0, -20), textcoords="offset points", ha="center", fontsize=13)
ax.set_xlabel("Transient tool failure rate (%)"); ax.set_ylabel("Tasks completed (%)"); ax.set_ylim(0, 108); ax.set_xticks([0, 5, 10, 20])
ax.legend(frameon=False, loc="lower left"); ax.grid(True, color="#DDD"); clean(ax)
ax.set_title("300 runs per point; the model also emits a malformed call 10% of the time", fontsize=13)
save(fig, "fig_15_3_reliability")

fig, ax = canvas(13, 6.2, 124, 56)
box(ax, 2, 20, 26, 16, "MCP host / client (Muse agent, IDE, chat app)", C_BLUE, fs=13.5)
box(ax, 48, 34, 30, 12, "northwind-content server", C_TEAL, fs=14); box(ax, 48, 12, 30, 12, "other servers (calendar, CMS, files)", C_GREY, fs=13.5)
for t, y in [("search_guidelines", 44), ("check_copy", 38), ("list_schedule", 32)]: box(ax, 90, y - 3, 32, 5.2, t, C_AMBER, fs=12.5)
arrow(ax, (28.2, 30), (48, 40)); arrow(ax, (28.2, 26), (48, 18)); arrow(ax, (78.2, 40), (90, 40))
label(ax, 38, 47, "JSON-RPC over stdio or HTTP", fs=12.5, style="italic"); label(ax, 106, 22, "schedule_post is not exposed:\npublishing stays behind approval", fs=12.5)
label(ax, 62, 53, "Model Context Protocol: tools described once, usable by any client", fs=15, bold=True)
save(fig, "fig_15_4_mcp")

mm = MS["memory"]
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.plot(range(1, 41), mm["full"], lw=2.8, color=L_RED, label="full history"); ax.plot(range(1, 41), mm["summarized"], lw=2.8, color=L_TEAL, label="summary + last 4 turns")
ax.text(40, mm["full"][-1] + 60, f"{mm['full'][-1]:,}", ha="right", fontsize=13.5, fontweight="bold"); ax.text(40, mm["summarized"][-1] + 60, f"{mm['summarized'][-1]}", ha="right", fontsize=13.5, fontweight="bold")
ax.set_xlabel("Conversation turns"); ax.set_ylabel("Prompt tokens for history"); ax.legend(frameon=False); ax.grid(True, color="#DDD"); clean(ax)
save(fig, "fig_15_5_memory")

sb = MS["sandbox"]
fig, ax = canvas(13, 5.8, 124, 52)
label(ax, 62, 49, "Sandboxed code execution: five real tests", fs=15, bold=True)
for i, (k, v) in enumerate(sb.items()):
    y = 37 - i * 8.5; outcome = {"normal calculation": "ran: printed 45", "infinite loop": f"killed by CPU limit after {v['seconds']} s",
                                "memory bomb": "stopped: MemoryError", "huge file": "stopped: file too large", "read secrets from environment": "secret not visible (printed None)"}[k]
    box(ax, 4, y, 46, 7, k, C_GREY, fs=13.5); box(ax, 56, y, 64, 7, outcome, C_GREEN if i == 0 else C_TEAL, fs=13.5)
save(fig, "fig_15_6_sandbox")
print("ok")
