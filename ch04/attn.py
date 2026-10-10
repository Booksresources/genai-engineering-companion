import sys, json, math, torch, torch.nn.functional as F
sys.path.insert(0, "/home/claude")
from mini_gpt import GPT, GPTConfig, apply_rope
torch.manual_seed(0); res = {}
# 1) Attention by hand: 4 tokens, 3-dimensional queries/keys/values
toks = ["The", "cat", "sat", "down"]
Q = torch.tensor([[1., 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1]]); K = torch.tensor([[1., 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
V = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
scores = Q @ K.T / math.sqrt(3)
mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), 1)          # a token may not look at later tokens
w = F.softmax(scores.masked_fill(mask, float("-inf")), -1); out = w @ V
res["manual"] = {"tokens": toks, "scores": scores.tolist(), "weights": w.tolist(), "out": out.tolist()}
print("weights:\n", w.round(decimals=2)); print("output for 'down':", out[3].round(decimals=2).tolist())
# 2) Attention maps from the trained Shakespeare model
text = open("/home/claude/tiny.txt").read(); chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
m = GPT(GPTConfig(vocab_size=65)); m.load_state_dict(torch.load("/home/claude/ckpt.pt", weights_only=False)["model"]); m.eval()
prompt = "ROMEO:\nBut soft, what light"
ids = torch.tensor([[stoi[c] for c in prompt]]); T = ids.shape[1]
maps = []
with torch.no_grad():
    x = m.embed(ids); cos, sin = m.cos[:T], m.sin[:T]
    for blk in m.blocks:
        a = blk.attn; h = blk.norm1(x)
        q = apply_rope(a.wq(h).view(1, T, a.n_heads, a.hd).transpose(1, 2), cos, sin)
        k = apply_rope(a.wk(h).view(1, T, a.n_kv, a.hd).transpose(1, 2), cos, sin).repeat_interleave(a.n_heads // a.n_kv, 1)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(a.hd)
        att = F.softmax(att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")), -1)
        maps.append(att[0].tolist()); x = blk(x, cos, sin)
res["maps"] = maps; res["prompt"] = prompt
json.dump(res, open("attn.json", "w")); print("layers", len(maps), "heads", len(maps[0]))
