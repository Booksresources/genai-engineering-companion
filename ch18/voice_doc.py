import asyncio, json, re, random, time
res = {}
# ---------- 18.2 voice pipeline: sequential versus streamed (sentence-level) cascade, with simulated component timings ----------
ASR_FINAL, LLM_TTFT, LLM_TOK_S, TTS_FIRST, TTS_RT = 0.30, 0.45, 40.0, 0.25, 0.2     # seconds; tokens/s; TTS real-time factor
REPLY = "Sure. The TrailMug keeps drinks hot for six hours. It fits standard cup holders. Want me to add it to the post?".split(" ")
async def llm_stream():
    await asyncio.sleep(LLM_TTFT)
    for w in REPLY: await asyncio.sleep(1 / LLM_TOK_S); yield w
async def tts(sentence):
    await asyncio.sleep(TTS_FIRST); return len(sentence.split()) * 0.3 * TTS_RT
async def sequential():
    t0 = time.perf_counter(); await asyncio.sleep(ASR_FINAL); words = [w async for w in llm_stream()]
    await tts(" ".join(words)); return time.perf_counter() - t0
async def streamed():
    t0 = time.perf_counter(); await asyncio.sleep(ASR_FINAL); buf, first = [], None
    async for w in llm_stream():
        buf.append(w)
        if w.endswith((".", "?", "!")):                       # send each complete sentence to TTS immediately
            await tts(" ".join(buf)); first = first or time.perf_counter() - t0; buf = []
            break                                             # time to first audio is what the user feels
    return first
res["voice"] = {"sequential_first_audio_s": asyncio.run(sequential()), "streamed_first_audio_s": asyncio.run(streamed()),
                "components": {"asr_final": ASR_FINAL, "llm_ttft": LLM_TTFT, "llm_tokens_per_s": LLM_TOK_S, "tts_first_audio": TTS_FIRST}}
print("voice:", {k: round(v, 2) if isinstance(v, float) else v for k, v in res["voice"].items()})
# ---------- 18.3 document AI: extract invoice fields, validate, route to review ----------
rng = random.Random(0)
def invoice(i, corrupt=None):
    items = [(rng.choice(["TrailMug", "EmberGloves", "RidgeTent"]), rng.randint(1, 20), rng.choice([18.0, 42.5, 129.0])) for _ in range(rng.randint(1, 4))]
    total = sum(q * p for _, q, p in items); date = f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
    lines = [f"INVOICE NW-{1000+i}", f"Date: {date}"] + [f"{n} x{q} @ {p:.2f}" for n, q, p in items] + [f"TOTAL {total:.2f}"]
    text = "\n".join(lines)
    if corrupt == "digit": text = text.replace(f"TOTAL {total:.2f}", f"TOTAL {total + 9:.2f}")              # OCR misread in the total
    if corrupt == "date": text = text.replace(date, date.replace("-", "/", 1).replace("2026", "2O26"))       # letter O for zero
    if corrupt == "line": text = "\n".join(l for l in text.split("\n") if not l.startswith(items[0][0]) or items[0][0] == "_")  # dropped line
    return text, {"total": round(total, 2), "date": date}
def extract(text):
    """Stand-in for an LLM/VLM extraction call returning schema-shaped fields."""
    num = re.search(r"INVOICE (\S+)", text); d = re.search(r"Date: (\S+)", text); tot = re.search(r"TOTAL ([\d.]+)", text)
    items = [(n, int(q), float(p)) for n, q, p in re.findall(r"(\w+) x(\d+) @ ([\d.]+)", text)]
    return {"number": num.group(1) if num else None, "date": d.group(1) if d else None, "total": float(tot.group(1)) if tot else None, "items": items}
def validate_invoice(f):
    problems = []
    if not f["date"] or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", f["date"]): problems.append("bad date")
    if f["total"] is None or abs(sum(q * p for _, q, p in f["items"]) - f["total"]) > 0.01: problems.append("line items do not sum to total")
    return problems
counts = {"clean": [0, 0], "digit": [0, 0], "date": [0, 0], "line": [0, 0]}
for i in range(400):
    kind = ["clean", "digit", "date", "line"][i % 4]; text, truth = invoice(i, None if kind == "clean" else kind)
    f = extract(text); flagged = bool(validate_invoice(f)); correct = f["total"] == truth["total"] and f["date"] == truth["date"] and kind != "line"
    counts[kind][0] += flagged; counts[kind][1] += 1
res["documents"] = {k: v[0] / v[1] for k, v in counts.items()}; print("share routed to human review:", res["documents"])
res["example_invoice"] = invoice(7)[0]
json.dump(res, open("voice_doc.json", "w"), indent=1)
