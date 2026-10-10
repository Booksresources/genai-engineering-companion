import sys, json, re, torch, torch.nn.functional as F
sys.path.insert(0, "/home/claude")
from mini_gpt import GPT, GPTConfig
text = open("/home/claude/tiny.txt").read(); chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
vocab = set(re.findall(r"[a-z]+", text.lower()))
m = GPT(GPTConfig(vocab_size=65)); m.load_state_dict(torch.load("/home/claude/ckpt.pt", weights_only=False)["model"]); m.eval()

def next_token(logits, temperature=1.0, top_k=None, top_p=None, min_p=None, greedy=False, rep_penalty=None, history=None):
    if rep_penalty and history is not None:                     # discourage recently used tokens
        for row, h in zip(logits, history):
            idx = torch.tensor(list(set(h[-40:]))); row[idx] = torch.where(row[idx] > 0, row[idx] / rep_penalty, row[idx] * rep_penalty)
    if greedy: return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    probs = F.softmax(logits, -1)
    if top_k:
        kth = torch.topk(probs, top_k).values[:, [-1]]; probs = torch.where(probs < kth, 0, probs)
    if top_p:
        sp, si = probs.sort(descending=True); drop = (sp.cumsum(-1) - sp) > top_p
        probs = probs.scatter(-1, si, sp.masked_fill(drop, 0))
    if min_p:
        probs = torch.where(probs < min_p * probs.max(-1, keepdim=True).values, 0, probs)
    return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1)

@torch.no_grad()
def run(n=12, L=240, **kw):
    torch.manual_seed(0); x = torch.tensor([[stoi[c] for c in "ROMEO:\n"]] * n)
    for _ in range(L):
        logits, _ = m(x[:, -128:]); x = torch.cat([x, next_token(logits[:, -1].clone(), history=x.tolist(), **kw)], 1)
    outs = ["".join(chars[i] for i in r[7:]) for r in x.tolist()]
    words = [w for o in outs for w in re.findall(r"[a-z]+", o.lower())]
    distinct = sum(len({o[i:i + 8] for i in range(len(o) - 8)}) / (len(o) - 8) for o in outs) / len(outs)
    return {"real_words": sum(w in vocab for w in words) / len(words), "distinct8": distinct, "example": outs[0][:150]}
configs = {"greedy": {"greedy": True}, "greedy + repetition penalty 1.3": {"greedy": True, "rep_penalty": 1.3},
           "temperature 0.7": {"temperature": 0.7}, "temperature 1.0": {"temperature": 1.0}, "temperature 1.4": {"temperature": 1.4},
           "top-k 10": {"top_k": 10}, "top-p 0.9": {"top_p": 0.9}, "temperature 1.4 + top-p 0.9": {"temperature": 1.4, "top_p": 0.9},
           "temperature 1.4 + min-p 0.1": {"temperature": 1.4, "min_p": 0.1}}
res = {k: run(**v) for k, v in configs.items()}
for k, v in res.items(): print(f"{k:32s} real {v['real_words']:.3f} distinct {v['distinct8']:.3f}")
print(repr(res["greedy"]["example"][:120])); print(repr(res["temperature 1.4"]["example"][:120]))
json.dump(res, open("decode.json", "w"), indent=1)
