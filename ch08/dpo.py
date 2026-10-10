import sys, json, random, time, torch, torch.nn.functional as F
from common8 import *
P = json.load(open("prefs.json"))["pairs"]
def dpo(beta=0.1, nll=0.0, steps=80, lr=1e-3, seed=0):
    torch.manual_seed(seed); rng = random.Random(seed)
    policy, ref = sft_model(), sft_model(); ref.eval()
    params = [p for p in policy.parameters() if p.requires_grad]            # only the LoRA adapters train
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0); hist = []
    for s in range(steps):
        b = rng.sample(P, 16); rows = [{"product": x["product"], "feature": x["feature"]} for x in b]
        lc = seq_logprob(policy, rows, [x["chosen"] for x in b]); lr_ = seq_logprob(policy, rows, [x["rejected"] for x in b])
        with torch.no_grad():
            rc = seq_logprob(ref, rows, [x["chosen"] for x in b]); rr = seq_logprob(ref, rows, [x["rejected"] for x in b])
        margin = (lc - rc) - (lr_ - rr)                                       # implicit reward difference / beta
        loss = -F.logsigmoid(beta * margin).mean()
        if nll: loss = loss - nll * (lc / torch.tensor([len(x["chosen"]) + 1 for x in b])).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if s % 10 == 0: hist.append((s, loss.item(), (lc - rc).mean().item(), (lr_ - rr).mean().item()))
    with torch.no_grad():
        rows = [{"product": x["product"], "feature": x["feature"]} for x in P]
        dc = (seq_logprob(policy, rows, [x["chosen"] for x in P]) - seq_logprob(ref, rows, [x["chosen"] for x in P])).mean().item()
        dr = (seq_logprob(policy, rows, [x["rejected"] for x in P]) - seq_logprob(ref, rows, [x["rejected"] for x in P])).mean().item()
    acc, outs = pass_rate(policy, TEST)
    return {"test_pass": acc, "chosen_logratio": dc, "rejected_logratio": dr, "hist": hist, "examples": [o[0] for o in outs[:3]],
            "mean_len": sum(len(o[0]) for o in outs) / len(outs)}
if __name__ == "__main__":
    name = sys.argv[1]; t0 = time.time()
    r = {"dpo": lambda: dpo(), "dpo_nll": lambda: dpo(nll=1.0), "dpo_low_lr": lambda: dpo(lr=2e-4), "dpo_nll_low_lr": lambda: dpo(nll=1.0, lr=2e-4)}[name]()
    r["seconds"] = time.time() - t0
    try: allr = json.load(open("dpo.json"))
    except FileNotFoundError: allr = {}
    allr[name] = r; json.dump(allr, open("dpo.json", "w"), indent=1)
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "hist"})
