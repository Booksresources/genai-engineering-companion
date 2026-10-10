"""A ReAct-style agent loop with schema validation, retries, a step budget, and human confirmation."""
import json, random, time, copy
from tools import TOOLS, VALIDATORS, run_tool, policy, SCHEDULE

class ScriptedModel:
    """Stand-in for an LLM's tool-calling decisions on the scheduling task. With `noise`, it sometimes emits malformed calls,
    as real models occasionally do; a real deployment replaces `next_action` with an LLM call that returns the same JSON."""
    def __init__(s, product="TrailMug", features=("keeps drinks hot for 6 hours",), day="2026-10-13", noise=0.0, seed=0):
        s.p, s.f, s.day, s.noise, s.rng = product, list(features), day, noise, random.Random(seed)
    def next_action(s, history):
        done = [h["tool"] for h in history if h.get("ok")]
        post = {"product": s.p, "headline": "Hot coffee, cold trails", "body": f"The {s.p} keeps drinks hot for 6 hours.", "hashtags": ["#Northwind", "#TrailMug"], "features": s.f}
        if "search_guidelines" not in done: call = ("search_guidelines", {"question": "What is the hashtag limit for social posts?"})
        elif "check_copy" not in done: call = ("check_copy", post)
        elif "list_schedule" not in done: call = ("list_schedule", {"start": s.day, "end": s.day})
        elif "schedule_post" not in done:
            busy = next(h["result"] for h in history if h["tool"] == "list_schedule" and h.get("ok"))
            slot = next(f"{s.day}T{hh:02d}:00" for hh in (9, 10, 11, 12) if f"{s.day}T{hh:02d}:00" not in busy)
            call = ("schedule_post", {"when": slot, "post": {k: post[k] for k in ("headline", "body", "hashtags")}})
        else: return {"final": f"Scheduled: {[h['result'] for h in history if h['tool'] == 'schedule_post' and h.get('ok')][-1]}"}
        name, args = call
        if s.rng.random() < s.noise:                                       # a malformed call: wrong type or missing field
            args = dict(args); k = next(iter(args)); args[k] = 123 if isinstance(args[k], str) else None
        return {"tool": name, "args": args}

def run_agent(model, approve=lambda name, args: True, robust=True, fail_rate=0.0, max_steps=12, retries=2, seed=0):
    rng = random.Random(seed); history = []; tainted = False
    for step in range(max_steps):
        act = model.next_action(history)
        if "final" in act: return {"ok": True, "answer": act["final"], "steps": step, "history": history}
        name, args = act["tool"], act["args"]
        if robust:
            errs = [e.message for e in VALIDATORS[name].iter_errors(args)]
            if errs:                                                       # send the validation error back instead of executing
                history.append({"tool": name, "ok": False, "result": "invalid arguments: " + "; ".join(errs)}); continue
        decision = policy(name, tainted)
        if decision == "deny": history.append({"tool": name, "ok": False, "result": "denied by policy"}); return {"ok": False, "reason": "denied", "steps": step, "history": history}
        if decision == "confirm" and not approve(name, args):
            history.append({"tool": name, "ok": False, "result": "rejected by human"}); return {"ok": False, "reason": "rejected", "steps": step, "history": history}
        for attempt in range(retries + 1 if robust else 1):
            try:
                result = run_tool(name, args, fail_rate, rng); history.append({"tool": name, "args": args, "ok": "error" not in (result or {}), "result": result}); break
            except Exception as e:
                if attempt == (retries if robust else 0): history.append({"tool": name, "ok": False, "result": f"error: {e}"});
                if not robust: return {"ok": False, "reason": f"crashed: {type(e).__name__}", "steps": step, "history": history}
    return {"ok": False, "reason": "step budget exhausted", "steps": max_steps, "history": history}

if __name__ == "__main__":
    import tools; saved = dict(tools.SCHEDULE)
    out = run_agent(ScriptedModel())
    for h in out["history"]: print("TOOL", h["tool"], "->", json.dumps(h["result"])[:110])
    print("FINAL", out.get("answer"), "| steps", out["steps"])
    tools.SCHEDULE.clear(); tools.SCHEDULE.update(saved)
    res = {"trace": [{"tool": h["tool"], "result": json.dumps(h["result"])[:120]} for h in out["history"]], "final": out.get("answer")}
    # reliability: transient tool failures and malformed model calls
    grid = {}
    for robust in [False, True]:
        for fail in [0.0, 0.05, 0.1, 0.2]:
            ok = 0
            for trial in range(300):
                tools.SCHEDULE.clear(); tools.SCHEDULE.update(saved)
                r = run_agent(ScriptedModel(noise=0.1, seed=trial), robust=robust, fail_rate=fail, seed=1000 + trial); ok += r["ok"]
            grid[f"{'robust' if robust else 'naive'}|{fail}"] = ok / 300
    print(grid); res["reliability"] = grid
    tools.SCHEDULE.clear(); tools.SCHEDULE.update(saved)
    r = run_agent(ScriptedModel(), approve=lambda n, a: False); res["human_rejects"] = r["reason"]; print("human rejects ->", r["reason"])
    json.dump(res, open("agent.json", "w"), indent=1)
