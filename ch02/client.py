"""A minimal, dependency-free client for OpenAI-compatible chat APIs with retries and cost tracking."""
import json, os, random, time, urllib.request, urllib.error

class ChatClient:
    def __init__(self, base_url, api_key, model, max_retries=5, timeout=30):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.max_retries, self.timeout, self.log = max_retries, timeout, []

    def chat(self, messages, **params):
        body = json.dumps({"model": self.model, "messages": messages, **params}).encode()
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                    self.log.append({"attempt": attempt, "status": 200, "t": t0}); return out
            except urllib.error.HTTPError as e:
                self.log.append({"attempt": attempt, "status": e.code, "t": t0})
                if e.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise                                              # do not retry auth or bad requests
                retry_after = float(e.headers.get("Retry-After", 0) or 0)
                wait = max(retry_after, min(30, 0.5 * 2 ** attempt)) * (1 + 0.1 * random.random())   # backoff + jitter
                time.sleep(wait)

PRICES = {"in": 1.0, "out": 4.0}          # illustrative USD per million tokens
def cost_of(response):
    u = response["usage"]; return (u["prompt_tokens"] * PRICES["in"] + u["completion_tokens"] * PRICES["out"]) / 1e6
