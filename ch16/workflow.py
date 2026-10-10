"""A small durable workflow engine: checkpointed steps, retries, parallel workers, critic loop, and human approval."""
import json, os, random, time, concurrent.futures as cf, sys
sys.path.insert(0, "/home/claude/gb/muse/src")
from muse.rag.brand import ask
from muse.prompts.library import check_copy

class NeedsApproval(Exception): pass
class Workflow:
    def __init__(s, name, state_dir="state"):
        s.path = os.path.join(state_dir, f"{name}.json"); os.makedirs(state_dir, exist_ok=True)
        s.state = json.load(open(s.path)) if os.path.exists(s.path) else {"done": {}, "log": []}
    def step(s, key, fn, *args, retries=2, backoff=0.01):
        if key in s.state["done"]:                                   # durable: completed steps are never re-run
            s.state["log"].append(f"skip {key}"); return s.state["done"][key]
        for attempt in range(retries + 1):
            try:
                out = fn(*args); break
            except NeedsApproval: raise
            except Exception as e:
                s.state["log"].append(f"retry {key}: {type(e).__name__}")
                if attempt == retries: s._save(); raise
                time.sleep(backoff * 2 ** attempt)
        s.state["done"][key] = out; s.state["log"].append(f"ran {key}"); s._save(); return out
    def _save(s): json.dump(s.state, open(s.path, "w"), indent=1)

# ---- step functions (stand-ins for LLM calls, with simulated latency and failures)
QUESTIONS = {"TrailMug": ["What is the hashtag limit for social posts?", "Which hex value is Fjord Green?", "What do waterproof claims require?",
                          "How should warranty language be phrased?"]}
def plan(product): time.sleep(0.05); return QUESTIONS[product]
def research(q, fail=0.0, rng=None):
    time.sleep(0.2)                                                   # one worker = one LLM/RAG call
    if rng and rng.random() < fail: raise TimeoutError("worker timeout")
    return ask(q)
def draft(product, findings, attempt=0):
    body = f"The {product} keeps drinks hot for 6 hours." if attempt else f"The {product} is the best ever mug! {product} keeps drinks hot for 12 hours."
    return {"headline": "Hot coffee, cold trails", "body": body, "hashtags": ["#Northwind", "#TrailMug"]}
def critic(product, post): return check_copy(post, product, ["keeps drinks hot for 6 hours"])
def approve(post, decision):
    if decision is None: raise NeedsApproval("waiting for a human")
    return decision

def campaign(name, product="TrailMug", crash_after=None, decision="approved", parallel=True, fail=0.0, seed=0):
    wf = Workflow(name); rng = random.Random(seed); t0 = time.time()
    qs = wf.step("plan", plan, product)
    if parallel:
        with cf.ThreadPoolExecutor(4) as ex:
            futs = {i: ex.submit(wf.step, f"research_{i}", research, q, fail, rng) for i, q in enumerate(qs)}
            findings = [futs[i].result() for i in range(len(qs))]
    else: findings = [wf.step(f"research_{i}", research, q, fail, rng) for i, q in enumerate(qs)]
    if crash_after == "research": raise SystemExit("simulated crash")
    post, attempt = wf.step("draft_0", draft, product, findings, 0), 0
    while (problems := wf.step(f"critic_{attempt}", critic, product, post)) and attempt < 2:
        attempt += 1; post = wf.step(f"draft_{attempt}", draft, product, findings, attempt)
    d = wf.step("approval", approve, post, decision)
    return {"post": post, "critic_rounds": attempt, "approval": d, "seconds": time.time() - t0, "log": wf.state["log"]}
