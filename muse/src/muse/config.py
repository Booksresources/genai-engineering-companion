import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    api_base: str
    api_key: str
    model: str
    monthly_budget_usd: float

def load_settings() -> Settings:
    missing = [k for k in ("MUSE_API_BASE", "MUSE_API_KEY", "MUSE_MODEL") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing settings: {', '.join(missing)} (see .env.example)")
    return Settings(os.environ["MUSE_API_BASE"], os.environ["MUSE_API_KEY"], os.environ["MUSE_MODEL"],
                    float(os.environ.get("MUSE_MONTHLY_BUDGET_USD", "200")))
