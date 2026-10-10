import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch02")
from matplotlib.ticker import FuncFormatter

# 2.1 Toolchain stack
fig, ax = canvas(13, 7.0, 120, 64)
layers = [("Your application: Muse (Chapter 2 skeleton onward)", C_GREEN), ("Model sources: Hugging Face Hub (open weights) and hosted model APIs", C_LAV),
          ("Libraries: Transformers, Diffusers, PEFT, TRL, datasets, vLLM", C_TEAL), ("PyTorch: tensors, autograd, nn.Module, optimizers", C_BLUE),
          ("Drivers and kernels: CUDA / ROCm / Metal, cuDNN, NCCL", C_AMBER), ("Hardware: CPU, local GPU, or cloud GPU", C_GREY)]
for i, (t, fc) in enumerate(layers):
    box(ax, 2, 52 - i * 9.6, 80, 8, t, fc, fs=13)
side = [("Environments and packages (uv, venv, conda)", C_GREY), ("Experiment tracking", C_GREY), ("Secrets and configuration", C_GREY), ("Cost tracking", C_GREY)]
for i, (t, fc) in enumerate(side):
    box(ax, 88, 48 - i * 12.5, 30, 10, t, fc, fs=12.5)
label(ax, 103, 61, "Cross-cutting tools", fs=13.5, bold=True)
label(ax, 42, 62, "The generative AI toolchain", fs=14.5, bold=True)
save(fig, "fig_2_1_toolchain")

# 2.2 Memory needed by model size and task
sizes = np.array([1, 3, 7, 13, 34, 70])
cases = [("Inference, 4-bit", lambda n: n * 0.55 * 1.2 + 1, L_TEAL, "o"), ("Inference, 16-bit", lambda n: n * 2 * 1.2 + 1, L_BLUE, "s"),
         ("QLoRA fine-tuning", lambda n: n * 0.6 + 3 + n * 0.15, L_AMBER, "^"), ("LoRA fine-tuning, 16-bit base", lambda n: n * 2.1 + 4 + n * 0.3, L_RED, "D")]
fig, ax = plt.subplots(figsize=(11, 6.0))
for name, f, col, mk in cases:
    ax.plot(sizes, f(sizes), marker=mk, ms=8, lw=2.6, color=col, label=name)
for g, lab in [(8, "8 GB laptop GPU"), (16, "16 GB"), (24, "24 GB (RTX 4090 class)"), (48, "48 GB"), (80, "80 GB (A100/H100)")]:
    ax.axhline(g, color="#777", ls="--", lw=1); ax.text(72, g, " " + lab, fontsize=11.5, color=BLACK, va="center")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([f"{s}B" for s in sizes])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); ax.set_ylim(0.8, 260)
ax.set_xlabel("Model size (parameters)"); ax.set_ylabel("Approximate GPU memory needed (GB, log)")
ax.set_xlim(0.85, 72); ax.legend(frameon=False, loc="upper left", fontsize=12); ax.grid(True, which="major", color="#DDD"); clean(ax)
ax.set_title("Rules of thumb; long contexts and large batches need more")
save(fig, "fig_2_2_memory")

# 2.3 Training loop + real result
d = json.load(open("/home/claude/gb/ch2/torch_basics.json")); g = json.load(open("/home/claude/gb/ch2/grid.json"))
fig = plt.figure(figsize=(13, 5.6))
ax = fig.add_axes([0.0, 0.0, 0.52, 1.0]); ax.set_xlim(0, 60); ax.set_ylim(0, 56); ax.axis("off")
st = [("1. Batch of data", C_GREY, 4, 40), ("2. Forward pass: model(x)", C_BLUE, 32, 40), ("3. Loss", C_AMBER, 32, 16), ("4. loss.backward()", C_ROSE, 4, 16)]
for t, fc, x, y in st: box(ax, x, y, 24, 10, t, fc, fs=12.5)
arrow(ax, (28.2, 45), (32, 45)); arrow(ax, (44, 40), (44, 26.2)); arrow(ax, (32, 21), (28.2, 21)); arrow(ax, (16, 26.2), (16, 40), ls="--")
label(ax, 18, 33, "5. optimizer.step(), zero_grad()", fs=12, ha="left", bold=True)
label(ax, 30, 4, "Repeat for many steps", fs=12.5, style="italic")
ax2 = fig.add_axes([0.58, 0.12, 0.40, 0.78])
G = np.array(g["grid"]); pr = np.array(g["pred"]).reshape(120, 120)
ax2.contourf(np.linspace(-1.1, 1.1, 120), np.linspace(-1.1, 1.1, 120), pr, levels=[0, 0.5, 1], colors=[C_BLUE, C_AMBER], alpha=0.8)
X = np.array(d["X"]); y = np.array(d["y"])
ax2.scatter(X[y == 0, 0], X[y == 0, 1], s=8, color=L_BLUE); ax2.scatter(X[y == 1, 0], X[y == 1, 1], s=8, color=L_AMBER)
ax2.set_title(f"Result after 400 steps: accuracy {100*d['hist'][-1][2]:.1f}%"); ax2.set_xticks([]); ax2.set_yticks([])
save(fig, "fig_2_3_loop")

# 2.4 Hub repository anatomy
fig, ax = canvas(13, 6.4, 120, 58)
label(ax, 60, 55.5, "What is inside a model repository on the Hugging Face Hub", fs=14.5, bold=True)
files = [("README.md (model card): intended use, training data, evaluation, limitations, license", C_GREEN),
         ("config.json: architecture hyperparameters", C_BLUE), ("model.safetensors: the weights (often sharded)", C_AMBER),
         ("tokenizer.json, tokenizer_config.json: tokenizer and chat template", C_TEAL), ("generation_config.json: default decoding settings", C_LAV)]
for i, (t, fc) in enumerate(files): box(ax, 2, 42 - i * 9.5, 70, 8, t, fc, fs=12.5, bold=False)
box(ax, 80, 34, 38, 16, "from_pretrained(\"org/model\")\ndownloads, caches, and loads", C_GREY, fs=12.5)
box(ax, 80, 10, 38, 16, "save_pretrained(\"dir\")\nwrites the same layout", C_GREY, fs=12.5)
arrow(ax, (72.2, 33), (80, 42)); arrow(ax, (80, 18), (72.2, 18))
save(fig, "fig_2_4_hub")

# 2.5 Retry timeline (real)
log = json.load(open("/home/claude/gb/ch2/retry_log.json"))
fig, ax = plt.subplots(figsize=(11, 3.8))
for e in log:
    col = L_RED if e["status"] != 200 else L_TEAL
    ax.add_patch(plt.Rectangle((e["t"], 0.3), 0.12, 0.4, color=col))
    ax.text(e["t"] + 0.06, 0.85, f"attempt {e['attempt']+1}\nHTTP {e['status']}", ha="center", fontsize=12.5, fontweight="bold", color=BLACK)
for a, b in zip(log, log[1:]):
    ax.annotate("", xy=(b["t"], 0.5), xytext=(a["t"] + 0.12, 0.5), arrowprops=dict(arrowstyle="->", lw=1.6, color=BLACK))
    ax.text((a["t"] + b["t"]) / 2 + 0.06, 0.38, f"wait {b['t'] - a['t']:.2f} s", ha="center", fontsize=12, color=BLACK)
ax.set_xlim(-0.2, log[-1]["t"] + 0.6); ax.set_ylim(0, 1.25); ax.set_yticks([]); ax.set_xlabel("Seconds since first attempt")
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.set_title("Real run of Listing 2.6 against the mock server: two rate limits, then success")
save(fig, "fig_2_5_retries")

# 2.6 Secrets flow
fig, ax = canvas(13, 5.0, 120, 44)
box(ax, 2, 22, 22, 12, ".env file (local only, in .gitignore)", C_AMBER, fs=12.5)
box(ax, 2, 4, 22, 12, "Secret manager (cloud, CI)", C_AMBER, fs=12.5)
box(ax, 34, 13, 22, 12, "Environment variables", C_BLUE, fs=12.5)
box(ax, 66, 13, 22, 12, "load_settings() validates at startup", C_TEAL, fs=12.5)
box(ax, 98, 13, 20, 12, "API client", C_GREEN, fs=12.5)
arrow(ax, (24.2, 28), (34, 21)); arrow(ax, (24.2, 10), (34, 17)); arrow(ax, (56.2, 19), (66, 19)); arrow(ax, (88.2, 19), (98, 19))
label(ax, 60, 39.5, "Secrets never appear in code, Git history, logs, prompts, or experiment configs", fs=13, bold=True)
save(fig, "fig_2_6_secrets")
print("ok")
