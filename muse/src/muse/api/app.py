"""Muse copywriting endpoint (Chapter 5)."""
import json, os, sys
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
sys.path.insert(0, "/home/claude/gb/ch5"); sys.path.insert(0, "/home/claude/gb/ch2")
from structured import structured_call, COPY_SCHEMA
from client import ChatClient, cost_of
from muse.config import load_settings

app = FastAPI(title="Muse")

class CopyRequest(BaseModel):
    product: str = Field(min_length=2, max_length=80)
    features: list[str] = Field(default_factory=list, max_length=8)
    channel: Literal["social", "email", "product_page"] = "social"

SYSTEM = ("You are Muse, Northwind Studio's copywriter. Voice: warm, outdoorsy, concise; no exclamation marks; "
          "never invent specifications. Reply only with JSON matching this schema: " + json.dumps(COPY_SCHEMA))
PRESETS = {"CREATIVE": {"temperature": 0.8, "top_p": 0.95, "max_tokens": 300}}

@app.post("/v1/copy")
def write_copy(req: CopyRequest):
    s = load_settings()
    client = ChatClient(s.api_base, s.api_key, s.model, max_retries=2)
    user = f"Channel: {req.channel}\nProduct: {req.product}\nFeatures: {'; '.join(req.features) or 'none given'}"
    try:
        data, log = structured_call(client, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], **PRESETS["CREATIVE"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"model call failed: {type(e).__name__}")
    return {"copy": data, "attempts": len(log), "prompt_version": "copy-v1"}
