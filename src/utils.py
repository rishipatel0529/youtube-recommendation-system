"""
utils.py
Environment loader for project configuration.
"""

import os
from dotenv import load_dotenv

def load_env():
    # Load .env file and ensure YT_API_KEY is set.
    load_dotenv()
    if not os.getenv("YT_API_KEY"):
        raise SystemExit("Missing YT_API_KEY in .env")