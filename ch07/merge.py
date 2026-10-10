import os, json, torch
from finetune import load_base, add_lora, LoRALinear, generate_copy, TEST, enc
m = load_base(); add_lora(m, r=8)
m.load_state_dict(torch.load("adapter.pt"), strict=False)                # base weights + small adapter
before = generate_copy(m, TEST[:20])
def merge_lora(model):
    """Fold B @ A into each base weight and replace LoRALinear with a plain Linear."""
    for blk in model.blocks:
        for parent in (blk.attn, blk.ffn):
            for name, mod in list(parent.named_children()):
                if isinstance(mod, LoRALinear):
                    mod.base.weight.data += mod.scale * mod.B.data @ mod.A.data
                    setattr(parent, name, mod.base)
    return model
merge_lora(m); after = generate_copy(m, TEST[:20])
torch.save(m.state_dict(), "merged.pt")
res = {"identical_outputs": before == after, "adapter_kb": os.path.getsize("adapter.pt") / 1024, "merged_kb": os.path.getsize("merged.pt") / 1024}
print(res); json.dump(res, open("merge.json", "w"))
