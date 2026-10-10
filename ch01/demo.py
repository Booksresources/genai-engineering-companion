import sys, json, torch
sys.path.insert(0, "/home/claude")
from mini_gpt import GPT, GPTConfig
text = open("/home/claude/tiny.txt").read(); chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
m = GPT(GPTConfig(vocab_size=65)); m.load_state_dict(torch.load("/home/claude/ckpt.pt", weights_only=False)["model"]); m.eval()
ctx = torch.tensor([[stoi[c] for c in "ROMEO:\n"]])
out = {"samples": []}
for seed in range(4):
    torch.manual_seed(seed)
    s = m.generate(ctx, 70, temperature=0.8, top_k=20)[0].tolist()
    out["samples"].append("".join(chars[i] for i in s[7:]))
s = m.generate(ctx, 70, temperature=1e-6, top_k=1)[0].tolist(); out["greedy"] = "".join(chars[i] for i in s[7:])
with torch.no_grad():
    probs = torch.softmax(m(ctx)[0][0, -1], -1)
top = torch.topk(probs, 8); out["next"] = [(chars[i], float(p)) for p, i in zip(top.values, top.indices)]
json.dump(out, open("demo.json", "w"), indent=1); print(json.dumps(out, indent=1))
