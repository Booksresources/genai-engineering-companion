import json, time, urllib.request
def stream_chat(base_url, api_key, model, messages, **params):
    """Yield text pieces as they arrive from an OpenAI-compatible streaming endpoint."""
    body = json.dumps({"model": model, "messages": messages, "stream": True,
                       "stream_options": {"include_usage": True}, **params}).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                yield {"usage": event["usage"]}
            for ch in event.get("choices", []):
                if ch["delta"].get("content"):
                    yield {"text": ch["delta"]["content"]}
