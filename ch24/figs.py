import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch24")
O = json.load(open("ops.json"))

fig, ax = canvas(13, 7.0, 124, 64)
label(ax, 62, 61, "The GenAIOps stack", fs=15, bold=True)
layers = [("Applications and workflows (Muse features)", C_GREEN), ("Gateway: routing, fallbacks, caching, rate and spend limits", C_BLUE), ("Registry: prompts, models, adapters, indexes, eval sets, with versions", C_LAV),
          ("Observability: traces, logs, metrics, quality monitors", C_TEAL), ("Evaluation and experimentation: release gates, A/B tests, feedback", C_AMBER), ("Controls: feature flags, kill switches, incident runbooks", C_ROSE)]
for i, (t, fc) in enumerate(layers): box(ax, 4, 50 - i * 9.5, 116, 7.8, t, fc, fs=14)
save(fig, "fig_24_1_stack")

tr = O["trace"]; root = [s for s in tr if s["parent"] is None][0]; kids = [s for s in tr if s["parent"]]
fig, ax = plt.subplots(figsize=(12, 5.2)); t0 = 0; order = ["plan"] + [f"research_{i}" for i in range(4)] + ["draft", "brand_check"]
ax.barh(len(order), root["ms"], left=0, color=C_GREY, edgecolor=EDGE); ax.text(root["ms"] + 8, len(order), f"campaign_request  {root['ms']:.0f} ms", va="center", fontsize=12.5, fontweight="bold")
for i, n in enumerate(order):
    s = next(k for k in kids if k["name"] == n); ax.barh(len(order) - 1 - i, max(s["ms"], 3), left=t0, color=C_TEAL if "research" in n else C_BLUE, edgecolor=EDGE)
    ax.text(t0 + max(s["ms"], 3) + 8, len(order) - 1 - i, f"{n}  {s['ms']:.0f} ms" + (f"  ${s['cost_usd']*1000:.2f}/1k" if s.get("cost_usd") else ""), va="center", fontsize=12); t0 += s["ms"]
ax.set_yticks([]); ax.set_xlabel("Milliseconds"); ax.set_xlim(0, 1500); clean(ax); ax.set_title("A real trace with version, token, and cost attributes on every span", fontsize=13)
save(fig, "fig_24_2_trace")

m = O["monitoring"]; d = np.array(m["daily"]); cs = np.array(m["cusum"])
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.0), sharex=True)
a1.plot(range(len(d)), 100 * d, "-o", ms=4, lw=1.6, color=L_BLUE); a1.axvline(40, color=L_RED, ls="--", lw=1.4); a1.text(40.5, 95.5, "regression starts (92% → 88%)", fontsize=12.5)
a1.axhline(85, color="#777", ls=":", lw=1.4); a1.text(1, 85.4, "fixed alarm at 85%: never fires", fontsize=12); a1.set_ylabel("Pass rate (%)"); a1.set_ylim(83, 97); clean(a1)
a2.plot(range(len(cs)), cs, lw=2.2, color=L_TEAL); a2.axhline(0.05, color=BLACK, ls="--", lw=1.2); a2.axvline(40, color=L_RED, ls="--", lw=1.4)
a2.text(m["cusum_first_alarm_after_change"] + 0.8, 0.055, f"CUSUM alarm: day {m['cusum_first_alarm_after_change']}", fontsize=12.5, fontweight="bold")
a2.set_xlabel("Day"); a2.set_ylabel("CUSUM statistic"); clean(a2)
fig.tight_layout(); save(fig, "fig_24_3_monitoring")

ab = O["ab"]; fp = ab["false_positives"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0))
a1.bar(["30% → 32%", "30% → 35%"], [ab["n_per_arm_30_to_32"], ab["n_per_arm_30_to_35"]], color=[C_ROSE, C_TEAL], edgecolor=EDGE, width=0.5)
for i, v in enumerate([ab["n_per_arm_30_to_32"], ab["n_per_arm_30_to_35"]]): a1.text(i, v + 150, f"{v:,}", ha="center", fontsize=14, fontweight="bold")
a1.set_ylabel("Users per arm"); a1.set_title("(a) Users needed per arm"); clean(a1)
ks = list(fp.keys()); a2.bar([f"{k} look{'s' if k != '1' else ''}" for k in ks], [100 * fp[k] for k in ks], color=[C_TEAL, C_AMBER, C_ROSE], edgecolor=EDGE, width=0.5)
for i, k in enumerate(ks): a2.text(i, 100 * fp[k] + 0.6, f"{100*fp[k]:.0f}%", ha="center", fontsize=14, fontweight="bold")
a2.axhline(5, color=BLACK, ls="--", lw=1.2); a2.set_ylabel("False positives (%)"); a2.set_title("(b) Peeking, no real effect"); clean(a2)
fig.tight_layout(w_pad=3); save(fig, "fig_24_4_ab")

fig, ax = canvas(13, 5.8, 124, 52)
st = [("production use", C_GREY), ("feedback, traces, failures", C_TEAL), ("triage: label and cluster", C_AMBER), ("new eval cases and training pairs", C_LAV), ("improve prompt, data, or model", C_BLUE), ("release gate", C_GREEN)]
for i, (t, fc) in enumerate(st):
    x = 1 + i * 20.6; box(ax, x, 20, 18.5, 14, t, fc, fs=13)
    if i < 5: arrow(ax, (x + 18.7, 27), (x + 20.6, 27))
arrow(ax, (112, 20), (10, 20), cs="arc3,rad=-0.25", ls="--"); label(ax, 62, 47, "The data flywheel", fs=15, bold=True)
save(fig, "fig_24_5_flywheel")

cf = O["cost_by_feature"]; ca = O["cache"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.4, 1]})
fs_ = list(cf.keys()); a1.barh(range(len(fs_))[::-1], [cf[f] for f in fs_], color=[C_BLUE, C_TEAL, C_AMBER, C_LAV], edgecolor=EDGE)
for i, f in enumerate(fs_): a1.text(cf[f] + 1, len(fs_) - 1 - i, f"${cf[f]:.0f}", va="center", fontsize=13)
a1.set_yticks(range(len(fs_))[::-1]); a1.set_yticklabels(fs_, fontsize=13); a1.set_xlabel("USD per month (illustrative prices)"); a1.set_title("(a) Cost by feature from traces"); a1.set_xlim(0, 75); clean(a1)
a2.pie([ca["hit_rate"], 1 - ca["hit_rate"]], labels=[f"served from cache\n{100*ca['hit_rate']:.0f}%", f"model calls\n{100*(1-ca['hit_rate']):.0f}%"], colors=[C_TEAL, C_ROSE],
       wedgeprops=dict(edgecolor=EDGE), textprops=dict(fontsize=12.5, color=BLACK)); a2.set_title("(b) Brand Q&A cache, 20,000 requests", fontsize=13)
fig.tight_layout(w_pad=2); save(fig, "fig_24_6_cost")
print("ok")
