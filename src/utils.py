import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    if not os.getenv("YT_API_KEY"):
        raise SystemExit("Missing YT_API_KEY in .env")
