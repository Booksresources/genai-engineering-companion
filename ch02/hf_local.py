import os, shutil, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, PreTrainedTokenizerFast
# Build a tiny model and tokenizer locally, save them in Hub format, and reload with the Auto* classes.
tok = PreTrainedTokenizerFast(tokenizer_file="/home/claude/ch3/bpe8k.json", eos_token="<|endoftext|>", pad_token="<|endoftext|>")
cfg = LlamaConfig(vocab_size=len(tok), hidden_size=128, intermediate_size=352, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
model = AutoModelForCausalLM.from_config(cfg)
shutil.rmtree("tiny-muse-lm", ignore_errors=True)
model.save_pretrained("tiny-muse-lm"); tok.save_pretrained("tiny-muse-lm")
print(sorted(os.listdir("tiny-muse-lm")))
m2 = AutoModelForCausalLM.from_pretrained("tiny-muse-lm"); t2 = AutoTokenizer.from_pretrained("tiny-muse-lm")
ids = t2("To be, or not to be", return_tensors="pt").input_ids
print("tokens:", ids.shape[1], "| params: %.2fM" % (sum(p.numel() for p in m2.parameters()) / 1e6))
out = m2.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=t2.pad_token_id)
print("untrained output:", repr(t2.decode(out[0][ids.shape[1]:])))
