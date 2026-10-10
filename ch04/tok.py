import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
text = open("/home/claude/tiny.txt").read(); n = int(0.9 * len(text)); train_txt, held = text[:n], text[n:]
open("train.txt", "w").write(train_txt)
res = {"compression": {}}
toks = {}
for V in [300, 1000, 4000, 16000]:
    tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
    tk.train(["train.txt"], trainers.BpeTrainer(vocab_size=V, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<|endoftext|>"]))
    ids = tk.encode(held).ids
    res["compression"][V] = {"chars_per_token": len(held) / len(ids), "actual_vocab": tk.get_vocab_size()}
    assert tk.decode(tk.encode(held[:2000]).ids) == held[:2000]           # lossless round trip
    toks[V] = tk
    print(V, round(len(held) / len(ids), 2))
examples = ["To be, or not to be, that is the question.", "Northwind TrailMug 2.0: only $24.99!", "unbelievably", "1234567"]
res["splits"] = {}
for V in [1000, 16000]:
    res["splits"][V] = []
    for s in examples:
        e = toks[V].encode(s)
        res["splits"][V].append({"text": s, "tokens": [toks[V].decode([i]) for i in e.ids]})
for e in res["splits"][16000]: print(len(e["tokens"]), e["tokens"])
json.dump(res, open("tok.json", "w"), indent=1)
toks[4000].save("bpe4k.json")
