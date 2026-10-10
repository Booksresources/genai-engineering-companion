import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch22")
I = json.load(open("infer.json")); I2 = json.load(open("infer2.json")); SP = json.load(open("spec.json"))

pdx = I["prefill_decode"]; Ls = list(pdx.keys())
fig, ax = plt.subplots(figsize=(11, 5.0)); x = np.arange(len(Ls)); w = 0.36
ax.bar(x - w / 2, [1000 * pdx[L]["prefill_s"] for L in Ls], w, color=C_TEAL, edgecolor=EDGE, label="prefill: whole prompt in one pass")
ax.bar(x + w / 2, [1000 * pdx[L]["token_by_token_s"] for L in Ls], w, color=C_ROSE, edgecolor=EDGE, label="decode: one token per pass")
for i, L in enumerate(Ls):
    ax.text(i - w / 2, 1000 * pdx[L]["prefill_s"] + 12, f"{1000*pdx[L]['prefill_s']:.0f}", ha="center", fontsize=12.5); ax.text(i + w / 2, 1000 * pdx[L]["token_by_token_s"] + 12, f"{1000*pdx[L]['token_by_token_s']:.0f}", ha="center", fontsize=12.5)
ax.set_xticks(x); ax.set_xticklabels([f"{L} tokens" for L in Ls]); ax.set_ylabel("Milliseconds"); ax.legend(frameon=False, loc="upper left"); clean(ax)
save(fig, "fig_22_1_prefill")

q = I["quant"]; names = list(q.keys())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0))
a1.bar(range(3), [q[n]["mb"] for n in names], color=[C_GREY, C_BLUE, C_TEAL], edgecolor=EDGE)
for i, n in enumerate(names): a1.text(i, q[n]["mb"] + 0.05, f"{q[n]['mb']:.2f} MB", ha="center", fontsize=13)
a1.set_xticks(range(3)); a1.set_xticklabels(["FP32", "INT8", "INT4 (sim.)"]); a1.set_ylabel("Model size (MB)"); a1.set_title("(a) Size"); clean(a1)
a2.bar(range(3), [q[n]["loss"] for n in names], color=[C_GREY, C_BLUE, C_TEAL], edgecolor=EDGE)
for i, n in enumerate(names): a2.text(i, q[n]["loss"] + 0.003, f"{q[n]['loss']:.3f}", ha="center", fontsize=13)
a2.set_ylim(1.49, 1.55); a2.set_xticks(range(3)); a2.set_xticklabels(["FP32", "INT8", "INT4 (sim.)"]); a2.set_ylabel("Validation loss"); a2.set_title("(b) Quality"); clean(a2)
fig.tight_layout(w_pad=3); save(fig, "fig_22_2_quant")

kv = I["kv"]; bt = I["batch"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0))
ns = list(kv.keys()); x = np.arange(len(ns)); w = 0.36
a1.bar(x - w / 2, [kv[n]["no_cache_s"] for n in ns], w, color=C_ROSE, edgecolor=EDGE, label="recompute everything")
a1.bar(x + w / 2, [kv[n]["cache_s"] for n in ns], w, color=C_TEAL, edgecolor=EDGE, label="KV cache")
a1.set_xticks(x); a1.set_xticklabels([f"{n} tokens" for n in ns]); a1.set_ylabel("Seconds"); a1.set_title("(a) KV cache"); a1.legend(frameon=False, loc="upper left"); clean(a1)
Bs = list(bt.keys()); a2.plot([int(b) for b in Bs], [bt[b]["tokens_per_s"] for b in Bs], "-o", ms=9, lw=2.6, color=L_BLUE)
for b in Bs: a2.annotate(f"{bt[b]['tokens_per_s']:.0f}", (int(b), bt[b]["tokens_per_s"]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=12.5)
a2.set_xscale("log", base=2); a2.set_xticks([1, 4, 16, 64]); a2.set_xticklabels(["1", "4", "16", "64"]); a2.set_xlabel("Batch size"); a2.set_ylabel("Tokens per second"); a2.set_title("(b) Batching (CPU)"); a2.set_ylim(0, 650); clean(a2)
fig.tight_layout(w_pad=3); save(fig, "fig_22_3_kv_batch")

ks = [k for k in SP if k != "wall"]
fig, ax = plt.subplots(figsize=(10.5, 5.0))
ax.plot([int(k) for k in ks], [SP[k]["tokens_per_call"] for k in ks], "-o", ms=9, lw=2.6, color=L_TEAL, label="tokens per target-model pass")
ax.plot([int(k) for k in ks], [100 * SP[k]["acceptance"] / 100 * 4 for k in ks], alpha=0)
for k in ks: ax.annotate(f"{SP[k]['tokens_per_call']:.2f}", (int(k), SP[k]["tokens_per_call"]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=12.5)
ax.axhline(1, color=BLACK, ls="--", lw=1.2); ax.text(8, 1.08, "normal decoding: 1", ha="right", fontsize=12.5)
ax.set_xlabel("Draft tokens proposed per step (k)"); ax.set_ylabel("Tokens per target pass"); ax.set_ylim(0, 4); ax.set_title("Outputs identical to the target model at every k", fontsize=13); clean(ax)
save(fig, "fig_22_4_speculative")

d = I2["diffusion"]
fig, ax = plt.subplots(figsize=(11, 5.4)); cols = [L_RED, L_BLUE, L_TEAL, L_PURPLE, L_AMBER]
for (n, v), col in zip(d.items(), cols):
    ax.scatter(v["seconds"], v["fd"], s=260, color=col, edgecolor=BLACK, zorder=5, label=f"{n}: FD {v['fd']:.2f}, {v['seconds']:.2f} s")
ax.set_xlabel("Seconds for 500 samples (CPU)"); ax.set_ylabel("Fréchet distance"); ax.set_ylim(1.5, 2.4); ax.set_xlim(0, 2.0)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=12); ax.grid(True, color="#DDD"); clean(ax)
save(fig, "fig_22_5_diffusion")

h = I2["halving"]; names = list(h.keys()); vals = [1000 * h[n] for n in names]
fig, ax = plt.subplots(figsize=(12, 5.4))
ax.barh(range(len(names))[::-1], vals, color=[C_ROSE, C_TEAL, C_AMBER, C_BLUE, C_GREEN], edgecolor=EDGE)
for i, v in enumerate(vals): ax.text(v + 4, len(names) - 1 - i, f"{v:.0f} ms", va="center", fontsize=13, fontweight="bold")
ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names, fontsize=12.5); ax.set_xlabel("Milliseconds per copy request (tiny GPT, CPU)"); ax.set_xlim(0, 340); clean(ax)
save(fig, "fig_22_6_halving")
print("ok")
