"""A tiny OpenAI-compatible mock server for testing clients offline.
It returns HTTP 429 (rate limited) for the first N requests, then a normal chat completion."""
import json, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    fail_first = 2; count = 0
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.count += 1
        if self.headers.get("Authorization") != "Bearer test-key":
            self.send_response(401); self.end_headers(); return
        if Handler.count <= Handler.fail_first:
            self.send_response(429); self.send_header("Retry-After", "1"); self.end_headers()
            self.wfile.write(b'{"error": {"message": "rate limit"}}'); return
        prompt = body["messages"][-1]["content"]
        reply = {"id": "chatcmpl-1", "object": "chat.completion", "model": body["model"],
                 "choices": [{"index": 0, "message": {"role": "assistant", "content": f"[mock] You said: {prompt[:40]}"},
                              "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": len(prompt.split()) + 12, "completion_tokens": 9, "total_tokens": len(prompt.split()) + 21}}
        data = json.dumps(reply).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(data)
def start(port=8765, fail_first=2):
    Handler.fail_first, Handler.count = fail_first, 0
    srv = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
