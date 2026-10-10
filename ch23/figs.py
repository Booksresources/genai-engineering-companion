import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch23")
L = json.load(open("load.json")); S = json.load(open("sims.json"))

fig, ax = canvas(13, 6.6, 124, 60)
label(ax, 62, 57, "Serving options", fs=15, bold=True)
rows = [("vLLM", "paged KV cache, continuous batching, OpenAI-compatible API, multi-LoRA", C_BLUE), ("SGLang", "high-throughput serving with prefix (radix) caching and structured output", C_TEAL),
        ("TGI / TensorRT-LLM", "Hugging Face server; NVIDIA-optimized kernels and engines", C_LAV), ("llama.cpp / Ollama", "quantized models on CPU, Apple silicon, edge devices", C_AMBER),
        ("Diffusion serving", "Diffusers or optimized pipelines behind a queue; Triton, Ray Serve, or custom workers", C_ROSE)]
for i, (t, d, fc) in enumerate(rows):
    y = 45 - i * 10.5; box(ax, 2, y, 30, 8.5, t, fc, fs=14); box(ax, 35, y, 87, 8.5, d, "white", fs=13, bold=False)
save(fig, "fig_23_1_engines")

fig, ax = canvas(13, 5.6, 124, 50)
st = [("code + lock file + model reference", C_GREY), ("container image (non-root, health check)", C_BLUE), ("registry (signed, versioned)", C_LAV), ("orchestrator: Kubernetes, managed or serverless GPUs", C_AMBER), ("probes, autoscaling, rollout", C_GREEN)]
for i, (t, fc) in enumerate(st):
    x = 1 + i * 24.6; box(ax, x, 18, 21.5, 16, t, fc, fs=13)
    if i < 4: arrow(ax, (x + 21.7, 26), (x + 24.6, 26))
label(ax, 62, 45, "From repository to running service", fs=15, bold=True)
label(ax, 62, 7, f"Real: Muse server ready in {L['startup_s']:.1f} s on CPU; GPU model servers often need minutes to load weights", fs=12.5, style="italic")
save(fig, "fig_23_2_deploy")

ld = L["load"]; cs = list(ld.keys())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0))
a1.plot([int(c) for c in cs], [ld[c]["rps"] for c in cs], "-o", ms=9, lw=2.6, color=L_BLUE); a1.set_xscale("log", base=2); a1.set_xticks([1, 4, 16, 64]); a1.set_xticklabels(["1", "4", "16", "64"])
a1.set_xlabel("Concurrent clients"); a1.set_ylabel("Requests per second"); a1.set_ylim(0, 400); a1.set_title("(a) Throughput saturates"); clean(a1)
a2.plot([int(c) for c in cs], [ld[c]["p50_ms"] for c in cs], "-o", ms=9, lw=2.6, color=L_TEAL, label="p50"); a2.plot([int(c) for c in cs], [ld[c]["p95_ms"] for c in cs], "-s", ms=9, lw=2.6, color=L_RED, label="p95")
a2.set_xscale("log", base=2); a2.set_xticks([1, 4, 16, 64]); a2.set_xticklabels(["1", "4", "16", "64"]); a2.set_xlabel("Concurrent clients"); a2.set_ylabel("Latency (ms)"); a2.legend(frameon=False); a2.set_title("(b) Latency grows with queueing"); clean(a2)
fig.tight_layout(w_pad=3); save(fig, "fig_23_3_load")

sc = S["scaling"]; tr = np.array(sc["autoscale"]["trace"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.4, 1]})
h = np.linspace(0, 24, 300); rate = 1.0 + 5.0 * np.exp(-((h - 14) / 3.5) ** 2)
a1.plot(h, rate * 3 / (4 * 0.7), color="#888", ls="--", lw=1.6, label="replicas needed (load)"); a1.step(tr[:, 0], tr[:, 1], where="post", color=L_BLUE, lw=2.4, label="autoscaled replicas")
a1.axhline(6, color=L_RED, lw=1.6, label="fixed 6"); a1.axhline(4, color=L_AMBER, lw=1.6, label="fixed 4")
a1.set_xlabel("Hour of day"); a1.set_ylabel("GPU replicas"); a1.legend(frameon=False, fontsize=11.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); a1.set_xlim(0, 24); clean(a1); a1.set_title("(a) A simulated day")
pols = list(sc.keys()); a2.bar(range(3), [sc[p]["gpu_hours"] for p in pols], color=[C_ROSE, C_AMBER, C_BLUE], edgecolor=EDGE)
for i, p in enumerate(pols): a2.text(i, sc[p]["gpu_hours"] + 3, f"{sc[p]['gpu_hours']:.0f} h\np95 {sc[p]['p95_s']:.0f} s", ha="center", fontsize=12)
a2.set_xticks(range(3)); a2.set_xticklabels(["fixed 6", "fixed 4", "auto"]); a2.set_ylabel("GPU-hours per day"); a2.set_ylim(0, 185); a2.set_title("(b) Cost and tail latency"); clean(a2)
fig.tight_layout(w_pad=2); save(fig, "fig_23_4_autoscale")

R = S["routing"]; names = list(R.keys())
fig, ax = plt.subplots(figsize=(11, 5.4))
for n, col in zip(names, [L_TEAL, L_RED, L_BLUE, L_PURPLE]):
    ax.scatter(R[n]["cost_per_request"], 100 * R[n]["quality"], s=300, color=col, edgecolor=BLACK, zorder=5)
    ax.annotate(n, (R[n]["cost_per_request"], 100 * R[n]["quality"]), xytext=(12, -4 if n != "router + fallback" else 8), textcoords="offset points", fontsize=13, fontweight="bold")
ax.set_xlabel("Relative cost per request (large model = 1.0)"); ax.set_ylabel("Answered well (%)"); ax.set_xlim(0, 1.3); ax.set_ylim(83, 97); ax.grid(True, color="#DDD"); clean(ax)
ax.set_title("Simulated: 25% hard requests; large model has 5% transient errors", fontsize=13)
save(fig, "fig_23_5_routing")

co = S["cost"]; us = list(co.keys())
fig, ax = plt.subplots(figsize=(10, 5.0))
ax.bar(range(len(us)), [co[u] for u in us], color=C_BLUE, edgecolor=EDGE, width=0.55)
for i, u in enumerate(us): ax.text(i, co[u] + 0.02, f"${co[u]:.2f}", ha="center", fontsize=13.5, fontweight="bold")
ax.set_xticks(range(len(us))); ax.set_xticklabels([f"{100*float(u):.0f}%" for u in us]); ax.set_xlabel("Average GPU utilization"); ax.set_ylabel("USD per 1k requests")
ax.set_title("$2.50/GPU-hour, 2,500 output tokens/s per GPU, 300 tokens per request (illustrative)", fontsize=12.5); clean(ax)
save(fig, "fig_23_6_cost")
print("ok")
