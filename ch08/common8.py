import sys, json, math, random, torch, torch.nn.functional as F
sys.path.insert(0, "/home/claude/gb/ch7")
import finetune as FT
from finetune import load_base, add_lora, enc, chars, prompt_for, brand_check, TRAIN, TEST
def sft_model():
    m = load_base(); add_lora(m, r=8); m.load_state_dict(torch.load("/home/claude/gb/ch7/adapter.pt"), strict=False); return m

@torch.no_grad()
def sample(m, rows, n=1, temperature=1.0, seed=0, max_new=80):
    """Batched sampling: prompts are grouped by length so no padding is needed."""
    m.eval(); g = torch.Generator().manual_seed(seed); out = [[None] * n for _ in rows]
    groups = {}
    for i, r in enumerate(rows): groups.setdefault(len(prompt_for(r["product"], r["feature"])), []).append(i)
    for L, idxs in groups.items():
        ids = torch.tensor([enc(prompt_for(rows[i]["product"], rows[i]["feature"])) for i in idxs for _ in range(n)])
        done = torch.zeros(len(ids), dtype=torch.bool); gen = [[] for _ in range(len(ids))]
        for _ in range(max_new):
            logits, _ = m(ids[:, -128:]); lg = logits[:, -1]
            nxt = lg.argmax(-1) if temperature == 0 else torch.multinomial(F.softmax(lg / temperature, -1), 1, generator=g).squeeze(-1)
            for k, t in enumerate(nxt.tolist()):
                if not done[k]:
                    if chars[t] == "\n": done[k] = True
                    else: gen[k].append(chars[t])
            ids = torch.cat([ids, nxt[:, None]], 1)
            if done.all(): break
        for k, i in enumerate(idxs for _ in [0]):
            pass
        k = 0
        for i in idxs:
            for j in range(n): out[i][j] = "".join(gen[k]); k += 1
    return out

def seq_logprob(m, rows, outs):
    """Sum of log-probabilities of each completion (plus newline) given its prompt."""
    xs, ys = [], []
    for r, o in zip(rows, outs):
        p, a = enc(prompt_for(r["product"], r["feature"])), enc(o + "\n")
        ids = (p + a)[:129]; lab = ([-100] * len(p) + a)[:129]
        ids += [0] * (129 - len(ids)); lab += [-100] * (129 - len(lab)); xs.append(ids[:-1]); ys.append(lab[1:])
    x, y = torch.tensor(xs), torch.tensor(ys); logits, _ = m(x)
    lp = torch.log_softmax(logits, -1).gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * (y != -100)).sum(-1)

def pass_rate(m, rows, temperature=0.0, n=1, seed=0):
    outs = sample(m, rows, n=n, temperature=temperature, seed=seed)
    return sum(brand_check(r, o[j])["pass"] for r, o in zip(rows, outs) for j in range(n)) / (len(rows) * n), outs
