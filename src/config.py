import os

def bool_from_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

USE_CF = os.getenv("USE_CF", "none").lower()  # "lightfm" or "none"
TRENDING_ONLY = bool_from_env("TRENDING_ONLY", False)
REGION = os.getenv("REGION", "US")
LONG_MIN_SEC = int(os.getenv("LONG_MIN_SEC", "180"))
