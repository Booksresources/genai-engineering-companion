"""Minimal experiment tracking and secret handling with only the standard library."""
import json, os, time, uuid, subprocess, pathlib

def load_dotenv(path=".env"):
    """Load KEY=VALUE lines into os.environ without overwriting existing variables."""
    p = pathlib.Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))

class Run:
    def __init__(self, name, config, root="runs"):
        self.id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.dir = pathlib.Path(root) / f"{name}-{self.id}"; self.dir.mkdir(parents=True)
        try: commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
        except Exception: commit = None
        safe = {k: v for k, v in config.items() if "key" not in k.lower()}          # never log secrets
        (self.dir / "config.json").write_text(json.dumps({"name": name, "git_commit": commit, **safe}, indent=1))
        self.f = open(self.dir / "metrics.jsonl", "a")
    def log(self, step=None, **metrics):
        self.f.write(json.dumps({"step": step, "time": time.time(), **metrics}) + "\n"); self.f.flush()
    def close(self): self.f.close()

if __name__ == "__main__":
    pathlib.Path(".env").write_text("MUSE_API_KEY=test-key\nMUSE_MODEL=mock-model\n")
    load_dotenv()
    run = Run("prompt-eval", {"model": os.environ["MUSE_MODEL"], "prompt_version": "v3", "api_key": os.environ["MUSE_API_KEY"]})
    for step, acc in enumerate([0.62, 0.71, 0.78]): run.log(step, accuracy=acc, cost_usd=0.004 * (step + 1))
    run.close()
    print(open(run.dir / "config.json").read()); print(open(run.dir / "metrics.jsonl").read().strip().splitlines()[-1])
