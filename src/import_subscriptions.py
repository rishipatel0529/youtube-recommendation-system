"""
import_subscriptions.py — Import YouTube subscriptions from CSV/JSON.

Purpose:
- Load environment and init DB.
- Parse subscription data from YouTube exports (CSV/JSON).
- Normalize channel IDs and titles.
- Insert/update subscriptions into local DB.
"""

import sys, time, json, csv, re
from pathlib import Path
from .utils import load_env
from . import store

def _norm_keys(d):
    # Normalize dict keys to lowercase alphanumeric (for flexible parsing).
    return {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in d.items()}

def _extract_from_dict(it):
    # Extract channel_id and title from a JSON item structure.
    ch_id = it.get("channelId") or (it.get("snippet") or {}).get("channelId") or (it.get("details") or {}).get("channelId")
    title = it.get("channelTitle") or (it.get("snippet") or {}).get("title") or (it.get("details") or {}).get("title") or ""
    return ch_id, title

def _extract_from_row(row):
    # Extract channel_id and title from a CSV row (after key normalization).
    n = _norm_keys(row)
    ch_id = n.get("channelid") or n.get("channel_id") or n.get("id") or n.get("ytchannelid")
    title = n.get("channeltitle") or n.get("title") or n.get("name")
    return ch_id, title or ""

def import_subs(path: Path):
    # Parse CSV/JSON subscriptions file and upsert into DB.
    load_env()
    store.init_db()
    rows, now = [], int(time.time())

    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ch_id, title = _extract_from_row(row)
                if ch_id:
                    rows.append({"channel_id": ch_id, "channel_title": title, "noted_at": now})
    else: # assume JSON
        text = path.read_text(encoding="utf-8").strip()
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("subscriptions") or data.get("items") or data.get("data") or []
        for it in data:
            ch_id, title = _extract_from_dict(it)
            if ch_id:
                rows.append({"channel_id": ch_id, "channel_title": title, "noted_at": now})

    if not rows:
        print("[subs] No subscriptions found.")
        return
    store.upsert_subscriptions(rows)
    print(f"[subs] Imported {len(rows)} subscriptions.")

def main():
    # CLI entry point: parse file path from args and import subs.
    if len(sys.argv) < 2:
        print("Usage: python -m src.import_subscriptions /path/to/subscriptions.(json|csv)")
        sys.exit(1)
    p = Path(sys.argv[1]).expanduser().resolve()
    if not p.exists():
        print("File not found:", p); sys.exit(2)
    import_subs(p)

if __name__ == "__main__":
    main()
