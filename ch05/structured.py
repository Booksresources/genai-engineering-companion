"""Structured output: validate model JSON against a schema, and repair by re-asking with the error."""
import json, sys
sys.path.insert(0, "/home/claude/gb/ch2")
from jsonschema import Draft202012Validator
from client import ChatClient

COPY_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "maxLength": 60},
        "body": {"type": "string", "maxLength": 400},
        "hashtags": {"type": "array", "items": {"type": "string", "pattern": "^#[A-Za-z0-9]+$"}, "maxItems": 3},
    },
    "required": ["headline", "body", "hashtags"], "additionalProperties": False,
}
validator = Draft202012Validator(COPY_SCHEMA)

def parse_and_validate(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, [f"invalid JSON: {e.msg}"]
    errors = [f"{'/'.join(map(str, err.path)) or '(root)'}: {err.message}" for err in validator.iter_errors(data)]
    return (data if not errors else None), errors

def structured_call(client, messages, max_attempts=3, **params):
    log = []
    for attempt in range(max_attempts):
        reply = client.chat(messages, **params)["choices"][0]["message"]["content"]
        data, errors = parse_and_validate(reply)
        log.append({"attempt": attempt + 1, "errors": errors})
        if data is not None:
            return data, log
        messages = messages + [{"role": "assistant", "content": reply},
                               {"role": "user", "content": "That response did not match the schema: " + "; ".join(errors)
                                + ". Reply with corrected JSON only."}]
    raise ValueError(f"no valid output after {max_attempts} attempts: {log[-1]['errors']}")

if __name__ == "__main__":
    from mock2 import start, H
    H.delay = 0; H.first_token_delay = 0
    bad1 = 'Sure! Here is your copy: {"headline": "Hot coffee, cold trails"}'
    bad2 = json.dumps({"headline": "Hot coffee, cold trails", "body": "The TrailMug keeps drinks hot for 6 hours.", "hashtags": ["#trail mug"]})
    good = json.dumps({"headline": "Hot coffee, cold trails", "body": "The TrailMug keeps drinks hot for 6 hours.", "hashtags": ["#TrailMug", "#Northwind"]})
    srv = start(port=8771, script=[bad1, bad2, good])
    client = ChatClient("http://127.0.0.1:8771/v1", "k", "mock")
    data, log = structured_call(client, [{"role": "system", "content": "Reply with JSON matching: " + json.dumps(COPY_SCHEMA)},
                                         {"role": "user", "content": "Social post for the TrailMug."}], temperature=0)
    for l in log: print(l)
    print("final:", data); srv.shutdown()
    json.dump(log, open("structured_log.json", "w"))
