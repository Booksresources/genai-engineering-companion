import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
path = "/home/claude/gb/ch2/tiny-muse-lm"              # replace with a downloaded instruct model, e.g. "org/model-instruct"
tok = AutoTokenizer.from_pretrained(path); model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
tok.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
                     "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")   # real models ship their own template
messages = [{"role": "system", "content": "You are Muse."}, {"role": "user", "content": "Tagline for the TrailMug?"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
ids = tok(prompt, return_tensors="pt").input_ids
t0 = time.time()
out = model.generate(ids, max_new_tokens=20, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id)
dt = time.time() - t0
print(f"prompt tokens {ids.shape[1]}, generated 20 tokens in {dt:.2f}s ({20/dt:.0f} tokens/s on CPU)")
