import json, time, sys
sys.path.insert(0, "/home/claude/gb/ch2")
from mock2 import start
from stream_client import stream_chat
from client import ChatClient
srv = start()
msgs = [{"role": "user", "content": "Write a tagline for the TrailMug."}]
t0 = time.time(); events = []; text = ""
for ev in stream_chat("http://127.0.0.1:8770/v1", "k", "mock", msgs, temperature=0.8, max_tokens=60):
    if "text" in ev: events.append(time.time() - t0); text += ev["text"]
    else: usage = ev["usage"]
print("streamed:", text); print(f"first token after {events[0]:.2f}s, last after {events[-1]:.2f}s, {len(events)} chunks, usage {usage}")
t0 = time.time(); r = ChatClient("http://127.0.0.1:8770/v1", "k", "mock").chat(msgs); t_block = time.time() - t0
print(f"non-streaming: whole reply after {t_block:.2f}s")
json.dump({"stream_times": events, "blocking": t_block}, open("stream.json", "w")); srv.shutdown()
