"""OpenAI-compatible mock server with streaming (server-sent events) and scripted replies."""
import json, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    script = []          # optional list of canned replies, consumed in order
    delay = 0.05         # seconds per streamed word (simulates generation speed)
    first_token_delay = 0.3
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        reply = H.script.pop(0) if H.script else f"Meet the trail-ready mug that keeps coffee hot for hours. {prompt[:30]}"
        usage = {"prompt_tokens": sum(len(m["content"].split()) for m in body["messages"]) + 8, "completion_tokens": len(reply.split())}
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            time.sleep(H.first_token_delay)
            for i, w in enumerate(reply.split(" ")):
                chunk = {"choices": [{"index": 0, "delta": {"content": (" " if i else "") + w}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode()); self.wfile.flush(); time.sleep(H.delay)
            self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n"); return
        time.sleep(H.first_token_delay + H.delay * len(reply.split()))
        out = {"choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}], "usage": usage}
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(json.dumps(out).encode())
def start(port=8770, script=None):
    H.script = list(script or [])
    s = HTTPServer(("127.0.0.1", port), H); threading.Thread(target=s.serve_forever, daemon=True).start(); return s
