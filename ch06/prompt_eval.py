"""A small prompt-evaluation harness and an automatic template search, run on the toy instruction model."""
import sys, json, itertools, random, time, torch
sys.path.insert(0, "/home/claude/p4")
from pt import new_model, generate, make_example

model = new_model(base=None); model.load_state_dict(torch.load("/home/claude/ch12/base_mask.pt")); model.eval()
TASKS = ["reverse", "upper", "sort", "add"]

def make_cases(n, seed):
    rng = random.Random(seed); cases = []
    for task in TASKS:
        for _ in range(n):
            p, a = make_example(task, rng); cases.append({"task": task, "input": p.split(": ", 1)[1], "expected": a})
    return cases

def run_prompt(template, cases):
    """template: dict task -> format string with {x}. Returns per-case pass/fail."""
    prompts = [template[c["task"]].format(x=c["input"]) for c in cases]
    outs = generate(model, prompts)
    return [{"task": c["task"], "ok": o == c["expected"], "out": o} for c, o in zip(cases, outs)]

def score(results, by_task=False):
    if not by_task: return sum(r["ok"] for r in results) / len(results)
    return {t: sum(r["ok"] for r in results if r["task"] == t) / sum(r["task"] == t for r in results) for t in TASKS}

if __name__ == "__main__":
    dev, test = make_cases(40, 1), make_cases(100, 2)
    hand = {"v1-natural": {t: f"Please {t} this: {{x}}" for t in TASKS},
            "v2-verb-colon": {t: f"{t.capitalize()}: {{x}}" for t in TASKS},
            "v3-training-format": {t: f"{t}: {{x}}" for t in TASKS}}
    res = {"hand": {}}
    for name, tpl in hand.items():
        r = run_prompt(tpl, test); res["hand"][name] = {"overall": score(r), "by_task": score(r, True)}
        print(f"{name:20s} {score(r):.3f}", {k: round(v, 2) for k, v in score(r, True).items()})
    # Automatic search: choose the best template per task on the dev set, then evaluate once on the test set
    verbs = {"reverse": ["reverse", "Reverse", "invert", "flip"], "upper": ["upper", "Upper", "uppercase", "caps"],
             "sort": ["sort", "Sort", "order", "alphabetize"], "add": ["add", "Add", "sum", "plus"]}
    seps, prefixes, suffixes = [": ", ":", " - ", " "], ["", "please "], ["", "."]
    t0 = time.time(); best, trace = {}, []
    for task in TASKS:
        dcases = [c for c in dev if c["task"] == task]; cands = []
        for v, s, p, q in itertools.product(verbs[task], seps, prefixes, suffixes):
            tpl = p + v + s + "{x}" + q
            acc = score(run_prompt({task: tpl}, dcases)); cands.append((acc, tpl))
        cands.sort(reverse=True); best[task] = cands[0][1]
        trace.append({"task": task, "n_candidates": len(cands), "best": cands[0], "worst": cands[-1], "median": cands[len(cands) // 2][0],
                      "all": [a for a, _ in cands]})
        print(task, "best", cands[0], "median", cands[len(cands) // 2][0], "worst", cands[-1])
    r = run_prompt(best, test)
    res["search"] = {"best_templates": best, "test_overall": score(r), "test_by_task": score(r, True), "trace": trace, "seconds": time.time() - t0}
    print("searched templates on held-out test:", round(score(r), 3), {k: round(v, 2) for k, v in score(r, True).items()}, f"{time.time()-t0:.0f}s")
    json.dump(res, open("prompt_eval.json", "w"), indent=1)
