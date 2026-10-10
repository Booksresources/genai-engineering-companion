import json, time, os
from mock_server import start
from client import ChatClient, cost_of
srv = start(fail_first=2)
os.environ.setdefault("MUSE_API_KEY", "test-key")                 # in real use, loaded from the environment or .env
client = ChatClient("http://127.0.0.1:8765/v1", os.environ["MUSE_API_KEY"], "mock-model")
t0 = time.time()
resp = client.chat([{"role": "user", "content": "Write a tagline for a waterproof hiking jacket."}], temperature=0.7)
print(resp["choices"][0]["message"]["content"])
print("attempts:", [(e["attempt"], e["status"], round(e["t"] - t0, 2)) for e in client.log])
print("usage:", resp["usage"], "cost: $%.6f" % cost_of(resp))
bad = ChatClient("http://127.0.0.1:8765/v1", "wrong-key", "mock-model")
try: bad.chat([{"role": "user", "content": "hi"}])
except Exception as e: print("bad key ->", type(e).__name__, getattr(e, "code", ""), "(not retried)")
json.dump([{**e, "t": e["t"] - t0} for e in client.log], open("retry_log.json", "w"))
srv.shutdown()
