import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch05")
ST = json.load(open("stream.json")); BU = json.load(open("budget.json")); SL = json.load(open("structured_log.json"))

# 5.1 API vs open weights
fig, ax = canvas(13, 9.0, 120, 80)
label(ax, 60, 77, "Proprietary APIs or open weights?", fs=16, bold=True)
box(ax, 30, 65, 42, 8, "Proprietary model API", C_BLUE, fs=15); box(ax, 76, 65, 42, 8, "Open-weight model you host", C_TEAL, fs=15)
rows = [("Best available quality", "Usually the frontier", "Close behind; improving fast"), ("Time to first result", "Minutes", "Hours to days"),
        ("Cost at low volume", "Low (pay per token)", "High (GPUs idle)"), ("Cost at high volume", "Can be very high", "Often lower per token"),
        ("Data control and residency", "Provider terms and regions", "Full control"), ("Customization", "Prompting; some fine-tuning", "Any fine-tuning, quantization, merging"),
        ("Operations burden", "Provider runs it", "You run serving, scaling, updates")]
for i, (k, a, b) in enumerate(rows):
    y = 55 - i * 8.8
    box(ax, 1, y, 27, 7.6, k, C_GREY, fs=13.5)
    box(ax, 30, y, 42, 7.6, a, "white", fs=13.5, bold=False); box(ax, 76, y, 42, 7.6, b, "white", fs=13.5, bold=False)
save(fig, "fig_5_1_choice")

# 5.2 request anatomy
fig, ax = canvas(13, 6.8, 120, 62)
label(ax, 28, 59, "Request", fs=15, bold=True); label(ax, 90, 59, "Response", fs=15, bold=True)
req = [("model: \"provider-model-2025-06\" (pinned)", C_GREY), ("messages[0] system: role, rules, format", C_LAV), ("messages[1] user: the task and its inputs", C_BLUE),
       ("messages[2] assistant / tool: earlier turns", C_GREY), ("temperature, top_p, max_tokens, stop, seed", C_AMBER), ("response_format / tools (structured output)", C_TEAL)]
for i, (t, fc) in enumerate(req): box(ax, 1, 48 - i * 8.6, 56, 7, t, fc, fs=13.5)
res = [("choices[0].message.content: generated text", C_GREEN), ("finish_reason: stop / length / tool_calls", C_AMBER), ("usage: prompt and completion tokens", C_ROSE), ("id, model, created: for logs and audits", C_GREY)]
for i, (t, fc) in enumerate(res): box(ax, 64, 44 - i * 10, 55, 8, t, fc, fs=13.5)
arrow(ax, (57.2, 30), (64, 30), lw=2.2)
save(fig, "fig_5_2_request")

# 5.3 streaming vs blocking (real)
t = np.array(ST["stream_times"])
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.scatter(t, np.ones_like(t) * 1, s=90, color=L_TEAL, edgecolor=BLACK, zorder=3, label="streamed chunks arrive")
ax.scatter([ST["blocking"]], [0], s=220, marker="D", color=L_RED, edgecolor=BLACK, zorder=3, label="blocking: whole reply arrives")
ax.annotate(f"first token {t[0]:.2f} s", (t[0], 1), xytext=(0, 22), textcoords="offset points", ha="center", fontsize=14, fontweight="bold")
ax.annotate(f"{ST['blocking']:.2f} s", (ST["blocking"], 0), xytext=(0, 18), textcoords="offset points", ha="center", fontsize=14, fontweight="bold")
ax.set_yticks([0, 1]); ax.set_yticklabels(["blocking", "streaming"]); ax.set_ylim(-0.6, 1.8); ax.set_xlim(0, 1.3)
ax.set_xlabel("Seconds after the request (mock server, 16 words)"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2); ax.grid(True, axis="x", color="#DDD"); clean(ax)
save(fig, "fig_5_3_streaming")

# 5.4 local options
fig, ax = canvas(13, 6.0, 120, 54)
opts = [("Transformers (Python)", "Full control, research, fine-tuning; slowest to serve", C_BLUE),
        ("llama.cpp (GGUF)", "Quantized models on CPU, Apple silicon, consumer GPUs", C_AMBER),
        ("Ollama", "One-command local models with an OpenAI-compatible API", C_TEAL),
        ("vLLM / SGLang", "High-throughput GPU serving with an OpenAI-compatible API", C_LAV)]
for i, (t, d, fc) in enumerate(opts):
    y = 41 - i * 12
    box(ax, 2, y, 30, 9, t, fc, fs=14.5); label(ax, 36, y + 4.5, d, fs=14, ha="left", width=60)
arrow(ax, (116, 6), (116, 50), lw=2); label(ax, 112, 28, "throughput and ease of serving", fs=13, ha="right", style="italic")
save(fig, "fig_5_4_local")

# 5.5 structured output loop
fig, ax = canvas(13, 6.2, 120, 56)
box(ax, 1, 32, 24, 16, "Call the model with the schema", C_BLUE, fs=13.5)
box(ax, 30, 34, 22, 12, "Parse JSON", C_AMBER, fs=14); box(ax, 59, 34, 24, 12, "Validate against JSON Schema", C_TEAL, fs=13.5)
box(ax, 92, 34, 26, 12, "Valid: return typed object", C_GREEN, fs=13.5)
box(ax, 44, 8, 32, 12, "Invalid: send the errors back and ask for corrected JSON (max 2 retries)", C_ROSE, fs=13)
arrow(ax, (25.2, 40), (30, 40)); arrow(ax, (52.2, 40), (59, 40)); arrow(ax, (83.2, 40), (92, 40))
arrow(ax, (71, 34), (66, 20.2)); arrow(ax, (44, 14), (12, 33.8), cs="arc3,rad=-0.3")
box(ax, 86, 4, 33, 22, "Real run: attempt 1 invalid JSON; attempt 2 bad hashtag pattern; attempt 3 valid", C_GREY, fs=12.5, bold=False)
save(fig, "fig_5_5_structured")

# 5.6 token budget (real)
parts = BU["parts"]; kept = {k for k, _ in BU["kept"]}
fig, ax = plt.subplots(figsize=(13, 5.6))
left = 0; cols = [C_LAV, C_ROSE, C_TEAL, C_AMBER, C_GREEN, C_GREY]
handles = []
for (name, n), col in zip(parts, cols):
    if name in kept:
        hb = ax.barh(0, n, left=left, color=col, edgecolor=EDGE, label=f"{name} ({n})")
        if n > 400: ax.text(left + n / 2, 0, str(n), ha="center", va="center", fontsize=13, fontweight="bold")
        left += n
ax.barh(0, 800, left=4096 - 800, color="white", edgecolor=EDGE, hatch="//", label="reserved for output (800)")
dn = [p for p in parts if p[0] not in kept][0]
ax.barh(-1, dn[1], left=0, color=C_GREY, edgecolor=EDGE, hatch="..", label=f"dropped: {dn[0]} ({dn[1]})")
ax.axvline(4096, color=BLACK, lw=2); ax.text(4060, 0.55, "context window: 4,096 tokens", ha="right", fontsize=13, fontweight="bold")
ax.set_yticks([0, -1]); ax.set_yticklabels(["sent", "dropped"]); ax.set_xlim(0, 4300); ax.set_ylim(-1.6, 0.9); ax.set_xlabel("Tokens"); clean(ax)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=12.5)
save(fig, "fig_5_6_budget")

# 5.7 Muse endpoint
fig, ax = canvas(13, 5.8, 120, 52)
st = [("POST /v1/copy\n{product, features, channel}", C_GREY), ("Validate request (Pydantic); 422 if invalid", C_AMBER), ("Build prompt: system voice + schema + inputs", C_LAV),
      ("structured_call: retries + schema repair", C_BLUE), ("Return copy JSON + attempts + prompt version; 502 on failure", C_GREEN)]
for i, (t, fc) in enumerate(st):
    x = 1 + i * 24; box(ax, x, 18, 21.5, 18, t, fc, fs=13)
    if i < 4: arrow(ax, (x + 21.7, 27), (x + 24, 27))
label(ax, 60, 46, "Muse copywriting endpoint (Chapter 5 build step)", fs=15, bold=True)
label(ax, 60, 7, "Tested offline against the mock server: valid copy, invalid input, and model failure.", fs=13.5, style="italic")
save(fig, "fig_5_7_endpoint")
print("ok")
