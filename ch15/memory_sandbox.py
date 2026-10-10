"""Memory: keep the context small with a rolling summary; sandbox: run untrusted code with limits."""
import json, os, resource, subprocess, sys, textwrap, time
sys.path.insert(0, "/home/claude/gb/ch5")
from budget import n_tokens
res = {}
# ---- memory: full history versus summary + recent turns
turns = [f"User: please draft a post about product number {i} with its key feature and check it against the guidelines. "
         f"Assistant: drafted the post, checked it, two problems fixed, scheduled for slot {i}." for i in range(40)]
full = [n_tokens(" ".join(turns[:k])) for k in range(1, 41)]
def summarized(k, keep=4):
    older = turns[:max(0, k - keep)]
    summary = f"Summary: {len(older)} earlier posts drafted, checked, and scheduled; preferences: no exclamation marks, mornings." if older else ""
    return n_tokens(summary + " " + " ".join(turns[max(0, k - keep):k]))
res["memory"] = {"full": full, "summarized": [summarized(k) for k in range(1, 41)]}
print("tokens after 40 turns: full", full[-1], "summary+recent", res["memory"]["summarized"][-1])
# ---- sandbox: subprocess with CPU, memory, file-size limits, no inherited secrets, and a wall-clock timeout
def run_sandboxed(code, timeout=2, mem_mb=200):
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout)); resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=timeout + 1, preexec_fn=limits,
                           env={"PATH": "/usr/bin"}, cwd="/tmp")
        return {"exit": p.returncode, "stdout": p.stdout.strip()[:60], "stderr": p.stderr.strip().splitlines()[-1][:60] if p.stderr.strip() else "", "seconds": round(time.time() - t0, 2)}
    except subprocess.TimeoutExpired:
        return {"exit": "timeout", "seconds": round(time.time() - t0, 2)}
os.environ["MUSE_API_KEY"] = "sk-secret"
cases = {"normal calculation": "print(sum(range(10)))", "infinite loop": "while True: pass",
         "memory bomb": "x = bytearray(10**9)", "huge file": "open('big.bin','wb').write(b'0'*50_000_000)",
         "read secrets from environment": "import os; print(os.environ.get('MUSE_API_KEY'))"}
res["sandbox"] = {k: run_sandboxed(v) for k, v in cases.items()}
for k, v in res["sandbox"].items(): print(k, v)
json.dump(res, open("memory_sandbox.json", "w"), indent=1)
