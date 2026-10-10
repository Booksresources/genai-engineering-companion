"""Muse campaign kit: one request -> brief, copy, image, voiceover, clip, using the book's small real models; times each stage."""
import sys, json, time, numpy as np, torch
for p in ["/home/claude/gb/ch7", "/home/claude/gb/ch9", "/home/claude/gb/ch10", "/home/claude/gb/ch11", "/home/claude/gb/ch12", "/home/claude/gb/muse/src"]: sys.path.insert(0, p)
timings, kit = {}, {}
def stage(name):
    class T:
        def __enter__(s): s.t = time.perf_counter()
        def __exit__(s, *a): timings[name] = time.perf_counter() - s.t
    return T()
brief_text = "Brief: launch the TrailMug for day hikers. Message: keeps drinks hot. Tone: warm, calm. Colors #2E5E4E #F4F1EA. Avoid prices. Deadline 2026-11-01."
with stage("1. brief extraction"):
    import re
    kit["brief"] = {"product": "TrailMug", "audience": re.search(r"for ([a-z ]+)\.", brief_text).group(1), "colors": re.findall(r"#[0-9A-F]{6}", brief_text),
                    "avoid": ["prices"], "deadline": re.search(r"\d{4}-\d{2}-\d{2}", brief_text).group(0)}
with stage("2. copy (fine-tuned LM, Ch. 7)"):
    import finetune as FT
    m = FT.load_base(); FT.add_lora(m, r=8); m.load_state_dict(torch.load("/home/claude/gb/ch7/adapter.pt"), strict=False)
    kit["copy"] = FT.generate_copy(m, [{"product": "Trail Mug", "feature": "keeps drinks hot"}])[0]
with stage("3. brand check (Ch. 6)"):
    from muse.prompts.library import check_copy
    kit["brand_problems"] = check_copy({"headline": "Hot coffee, cold trails", "body": kit["copy"].replace("Trail Mug", "TrailMug"), "hashtags": ["#Northwind"]}, "TrailMug", ["keeps drinks hot"])
with stage("4. image (text-to-image, Ch. 10)"):
    from t2i import TextUNet, generate
    from diff9 import schedule
    betas, abar = schedule("cosine", 200); tm = TextUNet(); tm.load_state_dict(torch.load("/home/claude/gb/ch10/t2i.pt")); tm.eval()
    img = generate(tm, ["a smiling face without glasses on white background"] * 4, abar, guidance=4.0); kit["images"] = 4
with stage("5. voiceover (TTS stand-in + normalize, Ch. 11)"):
    from audio import demo_signal
    def normalize_rms(x, target_dbfs=-20.0, peak_limit=0.95):               # Listing 11.7
        g = 10 ** (target_dbfs / 20) / max(np.sqrt((x ** 2).mean()), 1e-9); y = x * g; return y * min(1, peak_limit / np.abs(y).max())
    vo = normalize_rms(demo_signal(3.0)); kit["voiceover_seconds"] = 3.0
with stage("6. clip (image-to-video, Ch. 12)"):
    from video import Denoiser, sample, clips
    vm = Denoiser(True); vm.load_state_dict(torch.load("/home/claude/gb/ch12/video.pt")); vm.eval()
    clip = sample(vm, 1, abar, first_frame=clips(1, seed=3)[:, :, 0]); kit["clip_frames"] = int(clip.shape[2])
seq = sum(timings.values())
par = timings["1. brief extraction"] + max(timings["2. copy (fine-tuned LM, Ch. 7)"] + timings["3. brand check (Ch. 6)"], timings["4. image (text-to-image, Ch. 10)"],
                                            timings["5. voiceover (TTS stand-in + normalize, Ch. 11)"], timings["6. clip (image-to-video, Ch. 12)"])
out = {"timings": timings, "sequential_s": seq, "critical_path_s": par, "kit": kit}
print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in timings.items()}, indent=0)); print("sequential", round(seq, 2), "parallel critical path", round(par, 2)); print(kit["copy"], kit["brand_problems"])
json.dump(out, open("kit.json", "w"), indent=1)
