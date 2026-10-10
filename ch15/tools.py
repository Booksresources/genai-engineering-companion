"""Muse tools with JSON schemas, a permission policy, and a deterministic stand-in for the model's tool-calling decisions."""
import json, random, sys, datetime as dt
from jsonschema import Draft202012Validator
sys.path.insert(0, "/home/claude/gb/muse/src")
from muse.prompts.library import check_copy
from muse.rag.brand import ask

SCHEDULE = {"2026-10-13T09:00": "Ember Gloves launch post"}          # existing calendar
TOOLS = {
  "search_guidelines": {"description": "Answer a question from Northwind brand guidelines, with a citation.",
                        "parameters": {"type": "object", "properties": {"question": {"type": "string", "minLength": 5}}, "required": ["question"], "additionalProperties": False},
                        "risk": "read"},
  "check_copy": {"description": "Check a social post against Northwind brand rules. Returns a list of problems (empty if fine).",
                 "parameters": {"type": "object", "properties": {"product": {"type": "string"}, "headline": {"type": "string"}, "body": {"type": "string"},
                                "hashtags": {"type": "array", "items": {"type": "string"}}, "features": {"type": "array", "items": {"type": "string"}}},
                                "required": ["product", "headline", "body", "hashtags", "features"], "additionalProperties": False}, "risk": "read"},
  "list_schedule": {"description": "List scheduled posts between two ISO dates.",
                    "parameters": {"type": "object", "properties": {"start": {"type": "string", "format": "date"}, "end": {"type": "string", "format": "date"}},
                                   "required": ["start", "end"], "additionalProperties": False}, "risk": "read"},
  "schedule_post": {"description": "Schedule a post at an ISO datetime. Publishes publicly; requires human approval.",
                    "parameters": {"type": "object", "properties": {"when": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$"}, "post": {"type": "object"}},
                                   "required": ["when", "post"], "additionalProperties": False}, "risk": "write-public"},
}
VALIDATORS = {n: Draft202012Validator(t["parameters"]) for n, t in TOOLS.items()}

def run_tool(name, args, fail_rate=0.0, rng=None):
    if rng and rng.random() < fail_rate: raise TimeoutError(f"{name} timed out")              # simulated transient failure
    if name == "search_guidelines": return ask(args["question"])
    if name == "check_copy": return {"problems": check_copy({k: args[k] for k in ("headline", "body", "hashtags")}, args["product"], args["features"])}
    if name == "list_schedule": return {k: v for k, v in SCHEDULE.items() if args["start"] <= k[:10] <= args["end"]}
    if name == "schedule_post":
        if args["when"] in SCHEDULE: return {"error": "slot taken"}
        SCHEDULE[args["when"]] = args["post"]["headline"]; return {"scheduled": args["when"]}

# -------- permission policy --------
def policy(name, context_tainted):
    """allow / confirm / deny. Public writes always need a human; after reading untrusted content, they are denied outright."""
    risk = TOOLS[name]["risk"]
    if risk == "read": return "allow"
    if risk == "write-public": return "deny" if context_tainted else "confirm"
    return "deny"
