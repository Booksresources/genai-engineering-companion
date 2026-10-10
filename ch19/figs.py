import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch19")
E = json.load(open("evals.json")); G = json.load(open("gate.json"))

fig, ax = canvas(13, 6.6, 124, 60)
label(ax, 62, 57, "What to measure, and for whom", fs=15, bold=True)
lv = [("Capability", "does the model do the task? accuracy, rule pass rate, faithfulness", C_BLUE), ("Product", "do users succeed? acceptance, edits, task completion", C_TEAL),
      ("Safety", "harmful output, over-refusal, injection resistance (Ch. 20)", C_ROSE), ("Operations", "latency percentiles, cost per task, error rates (Ch. 22–24)", C_AMBER)]
for i, (t, d, fc) in enumerate(lv):
    y = 42 - i * 12; box(ax, 2, y, 26, 9.5, t, fc, fs=14.5); box(ax, 32, y, 90, 9.5, d, "white", fs=14, bold=False)
save(fig, "fig_19_1_layers")

rv = E["rouge_vs_rules"]; sc = np.array(rv["scatter"])
fig, ax = plt.subplots(figsize=(11, 5.2)); rng = np.random.default_rng(0)
for v, col, lab in [(0, L_RED, "fails brand rules"), (1, L_TEAL, "passes brand rules")]:
    s = sc[sc[:, 1] == v][:, 0]; ax.scatter(s, v + rng.uniform(-0.15, 0.15, len(s)), s=28, color=col, alpha=0.7, label=f"{lab} (n = {len(s)})")
ax.set_yticks([0, 1]); ax.set_yticklabels(["fails", "passes"]); ax.set_xlabel("ROUGE-L against a reference copy"); ax.set_xlim(0, 1.02)
ax.set_title(f"240 real outputs: correlation {rv['corr']:.2f}; {100*rv['high_rouge_but_fail']:.0f}% of outputs with ROUGE-L > 0.8 still fail", fontsize=13)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); clean(ax)
save(fig, "fig_19_2_rouge")

J = E["judge"]
fig, ax = plt.subplots(figsize=(11, 5.0))
vals = [100 * J["single_order_accuracy"], 100 * J["swap_accuracy_on_consistent"]]; labs = ["one order", "both orders,\nconsistent verdicts only"]
ax.bar(range(2), vals, color=[C_ROSE, C_TEAL], edgecolor=EDGE, width=0.5)
for i, v in enumerate(vals): ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=14, fontweight="bold")
ax.set_xticks(range(2)); ax.set_xticklabels(labs); ax.set_ylabel("Agreement with truth (%)"); ax.set_ylim(0, 110)
ax.set_title(f"Simulated judge with position bias: prefers the first answer {100*J['single_order_prefers_first']:.0f}% of the time;\nverdicts consistent under swapping {100*J['swap_consistent_share']:.0f}% of the time", fontsize=12.5); clean(ax)
save(fig, "fig_19_3_judge")

ci = E["ci"]; ns = list(ci.keys())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0))
p = E["rouge_vs_rules"]["pass_rate"]
for i, n in enumerate(ns):
    lo, hi = ci[n]; a1.plot([i, i], [100 * lo, 100 * hi], lw=6, color=L_BLUE); a1.text(i + 0.12, 100 * hi, f"{100*lo:.0f}–{100*hi:.0f}%", fontsize=12.5, va="center")
a1.axhline(100 * p, color=BLACK, ls="--", lw=1.2); a1.set_xticks(range(len(ns))); a1.set_xticklabels([f"{n}" for n in ns]); a1.set_xlim(-0.4, len(ns) - 0.2)
a1.set_xlabel("Items in the evaluation set"); a1.set_ylabel("95% interval (%)"); a1.set_title("(a) Interval for a 20% pass rate"); clean(a1)
pr = E["paired"]; a2.bar(range(2), [100 * pr["greedy"], 100 * pr["best_of_4"]], color=[C_GREY, C_TEAL], edgecolor=EDGE, width=0.5)
for i, v in enumerate([pr["greedy"], pr["best_of_4"]]): a2.text(i, 100 * v + 1.5, f"{100*v:.0f}%", ha="center", fontsize=14, fontweight="bold")
a2.set_xticks(range(2)); a2.set_xticklabels(["greedy", "best-of-4 +\nrule checker"]); a2.set_ylim(0, 60); a2.set_ylabel("Pass rate (%)")
a2.set_title(f"(b) Paired difference +{100*pr['diff']:.0f} pts\n(95% CI {100*pr['ci'][0]:.0f} to {100*pr['ci'][1]:.0f})", fontsize=13); clean(a2)
fig.tight_layout(w_pad=3); save(fig, "fig_19_4_stats")

pk = E["passk"]; ks = ["1", "2", "4"]
fig, ax = plt.subplots(figsize=(10, 5.0))
ax.plot([1, 2, 4], [100 * pk[k]["pass@k"] for k in ks], "-o", ms=9, lw=2.6, color=L_TEAL, label="pass@k: at least one of k tries passes")
ax.plot([1, 2, 4], [100 * pk[k]["pass^k"] for k in ks], "-s", ms=9, lw=2.6, color=L_RED, label="pass^k: all k tries pass")
for k in ks: ax.text(int(k), 100 * pk[k]["pass@k"] + 2.5, f"{100*pk[k]['pass@k']:.0f}%", ha="center", fontsize=12.5); ax.text(int(k), 100 * pk[k]["pass^k"] - 5, f"{100*pk[k]['pass^k']:.0f}%", ha="center", fontsize=12.5)
ax.set_xticks([1, 2, 4]); ax.set_xlabel("k (samples per product)"); ax.set_ylabel("% of 60 products"); ax.set_ylim(-8, 60); ax.legend(frameon=False, loc="upper left"); clean(ax)
save(fig, "fig_19_5_passk")

fig, ax = canvas(13, 6.6, 124, 60)
keys = list(G["base"]["results"].keys()); cfgs = [("base", "release candidate"), ("no_rewrite", "no query rewriting"), ("no_safeguards", "no agent safeguards")]
names = {"rag_recall_at_1": "RAG recall@1", "rag_recall_paraphrased": "RAG recall@1, paraphrased", "rag_abstains_unanswerable": "abstains on unanswerable",
         "brand_checker_catches_violations": "brand checker catches errors", "agent_completion_with_failures": "agent completion, tool failures"}
label(ax, 2, 55, "metric (threshold)", fs=13.5, bold=True, ha="left")
for j, (c_, lab) in enumerate(cfgs): label(ax, 71 + j * 19, 55, lab, fs=12.5, bold=True, width=12)
for i, k in enumerate(keys):
    y = 45 - i * 8; label(ax, 2, y, f"{names[k]} (≥ {G['base']['thresholds'][k]:.2f})", fs=12.5, ha="left")
    for j, (c_, lab) in enumerate(cfgs):
        v = G[c_]["results"][k]; ok = v >= G["base"]["thresholds"][k]
        box(ax, 63 + j * 19, y - 3, 16, 6, f"{v:.2f}", C_GREEN if ok else C_ROSE, fs=13)
label(ax, 2, 3, "release decision", fs=13.5, bold=True, ha="left")
for j, (c_, lab) in enumerate(cfgs): box(ax, 63 + j * 19, 0, 16, 6, "PASS" if G[c_]["passed"] else "BLOCKED", C_GREEN if G[c_]["passed"] else C_ROSE, fs=13)
save(fig, "fig_19_6_gate")
print("ok")
