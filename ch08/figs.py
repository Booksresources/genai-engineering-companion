import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch08")
PR = json.load(open("prefs.json")); DP = json.load(open("dpo.json")); BN = json.load(open("bon.json"))

# 8.1 imitation vs preference
fig, ax = canvas(12, 5.6, 120, 50)
label(ax, 30, 47, "SFT: imitate", fs=15, bold=True); label(ax, 91, 47, "Preferences: compare", fs=15, bold=True)
box(ax, 4, 26, 52, 12, "prompt → one demonstration of a good answer", C_BLUE, fs=14)
box(ax, 4, 8, 52, 12, "teaches what good output looks like", C_GREY, fs=14, bold=False)
box(ax, 64, 30, 25, 10, "answer A", C_GREEN, fs=14); box(ax, 93, 30, 25, 10, "answer B", C_ROSE, fs=14)
label(ax, 91, 35, ">", fs=22, bold=True)
box(ax, 64, 8, 54, 14, "teaches which of two plausible answers is better, including mistakes to avoid", C_GREY, fs=14, bold=False)
save(fig, "fig_8_1_imitation")

# 8.2 collecting preferences (real stats)
st = PR["stats"]
fig = plt.figure(figsize=(13, 5.6))
ax = fig.add_axes([0.0, 0.0, 0.55, 1.0]); ax.set_xlim(0, 66); ax.set_ylim(0, 56); ax.axis("off")
src = [("User feedback: thumbs, edits, choices between two drafts", C_LAV), ("Human annotators with guidelines", C_BLUE), ("AI judge or rule checker", C_TEAL)]
for i, (t, fc) in enumerate(src):
    box(ax, 1, 40 - i * 15, 30, 12, t, fc, fs=13.5); arrow(ax, (31.2, 46 - i * 15), (38, 28))
box(ax, 38, 21, 27, 14, "pairs: prompt, chosen, rejected", C_AMBER, fs=14)
a2 = fig.add_axes([0.62, 0.17, 0.36, 0.7])
vals = [st["all_pass"], st["mixed"], st["all_fail"]]
a2.bar(["all 4\npass", "mixed\n(→ pairs)", "all 4\nfail"], vals, color=[C_GREEN, C_AMBER, C_ROSE], edgecolor=EDGE)
for i, v in enumerate(vals): a2.text(i, v + 2, str(v), ha="center", fontsize=14, fontweight="bold")
a2.set_ylabel("Prompts (of 120)"); a2.set_title("Real: 4 samples per prompt,\njudged by the brand-rule checker", fontsize=14); a2.set_ylim(0, 110); clean(a2)
save(fig, "fig_8_2_collect")

# 8.3 DPO diagram
fig, ax = canvas(12, 6.0, 120, 54)
box(ax, 1, 30, 22, 12, "chosen answer", C_GREEN, fs=14); box(ax, 1, 10, 22, 12, "rejected answer", C_ROSE, fs=14)
box(ax, 32, 30, 26, 12, "policy (trainable)", C_BLUE, fs=14); box(ax, 32, 10, 26, 12, "reference (frozen SFT model)", C_GREY, fs=14)
for y in (36, 16): arrow(ax, (23.2, y), (32, 36)); arrow(ax, (23.2, y), (32, 16))
box(ax, 66, 18, 52, 18, "loss = −log σ( β [ (log π − log π_ref)(chosen) − (log π − log π_ref)(rejected) ] )", C_AMBER, fs=14)
arrow(ax, (58.2, 36), (66, 30)); arrow(ax, (58.2, 16), (66, 24))
label(ax, 60, 49, "Direct Preference Optimization: no reward model, no sampling during training", fs=15, bold=True)
save(fig, "fig_8_3_dpo")

# 8.4 DPO results (real)
keys = [("dpo", "DPO"), ("dpo_nll", "DPO\n+NLL"), ("dpo_nll_low_lr", "DPO+NLL\nlow LR")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
vals = [100 * st["sft_test_greedy"]] + [100 * DP[k]["test_pass"] for k, _ in keys]
a1.bar(range(4), vals, color=[C_GREY, C_ROSE, C_AMBER, C_TEAL], edgecolor=EDGE, width=0.6)
for i, v in enumerate(vals): a1.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=14, fontweight="bold")
a1.set_xticks(range(4)); a1.set_xticklabels(["SFT"] + [l for _, l in keys], fontsize=13); a1.set_ylabel("Pass rate (%)"); a1.set_ylim(0, 40)
a1.set_title("(a) Task result"); clean(a1)
x = np.arange(3); w = 0.36
a2.bar(x - w / 2, [DP[k]["chosen_logratio"] for k, _ in keys], w, color=C_GREEN, edgecolor=EDGE, label="chosen")
a2.bar(x + w / 2, [DP[k]["rejected_logratio"] for k, _ in keys], w, color=C_ROSE, edgecolor=EDGE, label="rejected")
a2.axhline(0, color=BLACK, lw=1); a2.set_xticks(x); a2.set_xticklabels([l for _, l in keys], fontsize=12.5)
a2.set_ylabel("Log-ratio (nats)"); a2.set_title("(b) Likelihood change"); a2.legend(frameon=False, loc="lower right"); clean(a2)
fig.tight_layout(w_pad=3); save(fig, "fig_8_4_dpo_results")

# 8.5 RLHF
fig, ax = canvas(12, 6.0, 120, 54)
st5 = [("1. SFT model", C_BLUE), ("2. Reward model trained on preference pairs", C_TEAL), ("3. RL (PPO): generate, score, update with KL penalty to SFT", C_AMBER)]
for i, (t, fc) in enumerate(st5):
    x = 2 + i * 40; box(ax, x, 22, 34, 16, t, fc, fs=14)
    if i < 2: arrow(ax, (x + 34.2, 30), (x + 40, 30))
label(ax, 60, 47, "Classic RLHF", fs=15, bold=True)
label(ax, 60, 9, "reward = reward model score − β · KL(policy ‖ SFT model)", fs=15, bold=True)
save(fig, "fig_8_5_rlhf")

# 8.6 RLVR
fig, ax = canvas(12, 5.8, 120, 52)
st6 = [("prompt with a checkable answer (math, code, rules)", C_GREY), ("sample several responses, often with long reasoning", C_BLUE), ("verifier: tests pass? answer correct? rules met?", C_TEAL), ("reinforce responses that pass (e.g. GRPO)", C_AMBER)]
for i, (t, fc) in enumerate(st6):
    x = 1 + i * 30; box(ax, x, 18, 26, 18, t, fc, fs=13.5)
    if i < 3: arrow(ax, (x + 26.2, 27), (x + 30, 27))
arrow(ax, (105, 18), (14, 18), cs="arc3,rad=-0.3", ls="--")
label(ax, 60, 47, "Reinforcement learning with verifiable rewards", fs=15, bold=True)
save(fig, "fig_8_6_rlvr")

# 8.7 best-of-n (real)
ns = [1, 2, 4, 8]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})
for k, col, mk in [("oracle (rule checker)", L_TEAL, "o"), ("model confidence", L_BLUE, "s"), ("length-biased judge", L_RED, "^")]:
    a1.plot(ns, [100 * BN[str(n)][k] for n in ns], marker=mk, ms=9, lw=2.6, color=col, label=k)
for n in ns: a1.annotate(f"{100*BN[str(n)]['oracle (rule checker)']:.0f}%", (n, 100 * BN[str(n)]["oracle (rule checker)"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=12.5)
a1.set_xscale("log", base=2); a1.set_xticks(ns); a1.set_xticklabels([str(n) for n in ns]); a1.set_xlabel("n (samples per prompt)"); a1.set_ylabel("Pass rate (%)")
a1.set_ylim(10, 70); a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12); a1.grid(True, color="#DDD"); a1.set_title("(a) Best-of-n"); clean(a1)
a2.plot(ns, [BN[str(n)]["mean length chosen by length judge"] for n in ns], "-^", ms=9, lw=2.6, color=L_RED)
a2.set_xscale("log", base=2); a2.set_xticks(ns); a2.set_xticklabels([str(n) for n in ns]); a2.set_xlabel("n"); a2.set_ylabel("Chosen length (chars)")
a2.set_title("(b) Length bias"); a2.grid(True, color="#DDD"); clean(a2)
fig.tight_layout(w_pad=3); save(fig, "fig_8_7_bon")
print("ok")
