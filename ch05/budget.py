"""Token budgeting: fit prompt parts into a context window, highest priority first."""
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/home/claude/gb/ch4/bpe4k.json")
def n_tokens(text): return len(tok.encode(text).ids)

def fit_to_budget(parts, context_window, reserve_output):
    """parts: list of (name, text, priority, required). Lower priority number = more important.
    Required parts must fit; optional parts are added whole, in priority order, while they fit."""
    budget = context_window - reserve_output; used, kept, dropped = 0, [], []
    for name, text, prio, required in sorted(parts, key=lambda p: p[2]):
        t = n_tokens(text)
        if required or used + t <= budget:
            used += t; kept.append((name, t))
        else:
            dropped.append((name, t))
    if used > budget:
        raise ValueError(f"required parts need {used} tokens but only {budget} are available")
    return kept, dropped, used, budget

if __name__ == "__main__":
    text = open("/home/claude/tiny.txt").read()
    parts = [("system prompt", text[:1200], 0, True), ("user question", text[5000:5300], 1, True),
             ("brand guide excerpt", text[10000:14000], 2, False), ("example 1", text[20000:21500], 3, False),
             ("example 2", text[30000:31500], 4, False), ("conversation history", text[40000:46000], 5, False)]
    for name, t, *_ in parts: print(f"{name:22s} {n_tokens(t):5d} tokens")
    kept, dropped, used, budget = fit_to_budget(parts, context_window=4096, reserve_output=800)
    print("kept:", kept); print("dropped:", dropped); print(f"used {used} of {budget}")
    import json; json.dump({"parts": [(n, n_tokens(t)) for n, t, *_ in parts], "kept": kept, "dropped": dropped, "used": used, "budget": budget}, open("budget.json", "w"))
