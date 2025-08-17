"""
import_takeout.py — Import YouTube watch history from Google Takeout.

Purpose:
- Parse Takeout JSON watch-history exports.
- Extract video IDs and timestamps.
- Filter for long-form videos (skip Shorts).
- Fetch metadata via YouTube API and insert into DB/CSV.
"""

import os, sys, json, time
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from time import sleep

from .utils import load_env
from . import store, youtube

# Configurable limits
MAX_IMPORT = int(os.getenv("MAX_IMPORT", "300")) # target long-form imports
SCAN_WINDOW = int(os.getenv("SCAN_WINDOW", "5000")) # rows to scan from end of file
MIN_DURATION_SEC = int(os.getenv("LONG_MIN_SEC", "180")) # long-form threshold

YOUTUBE_DOMAINS = ("youtube.com", "m.youtube.com", "www.youtube.com", "youtu.be")

def _extract_video_id(url: str):
    # Extract video ID from a YouTube URL (handles youtu.be and /watch).
    if not url:
        return None
    try:
        u = urlparse(url)
        if u.hostname not in YOUTUBE_DOMAINS:
            return None
        if u.hostname == "youtu.be":
            vid = u.path.strip("/").split("/")[0]
            return vid or None
        if u.path == "/watch":
            return parse_qs(u.query).get("v", [None])[0]
        return None # ignore other paths
    except Exception:
        return None

def _load_takeout_json(path: Path):
    # Load and normalize watch-history.json into [(timestamp, video_id)].
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for it in data:
        url = it.get("titleUrl") or ""
        vid = _extract_video_id(url)
        if not vid:
            continue
        ts_iso = it.get("time")
        try:
            from datetime import datetime, timezone
            ts = int(datetime.fromisoformat(ts_iso.replace("Z","+00:00")).timestamp())
        except Exception:
            ts = int(time.time())
        rows.append((ts, vid))
    rows.sort()
    return rows

def _chunk(lst, n):
    # Yield list chunks of size n.
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def import_watch_history(json_path: Path):
    # Main import logic: parse, dedup, filter, fetch metadata, upsert to DB.
    load_env()
    store.init_db()

    print(f"[import] Reading {json_path}")
    rows = _load_takeout_json(json_path)
    total_rows = len(rows)
    if not rows:
        print("[import] No watch entries found.")
        return

    # Dedup while preserving order
    seen = set()
    ordered = []
    for ts, vid in rows:
        if vid in seen:
            continue
        seen.add(vid)
        ordered.append((ts, vid))

    print(f"[import] Unique video IDs in file: {len(ordered)} (from {total_rows} rows)")

    # Limit scan window
    if len(ordered) > SCAN_WINDOW:
        ordered = ordered[-SCAN_WINDOW:]
        print(f"[import] Scanning the most recent {SCAN_WINDOW} unique IDs")

    print(f"[import] Fetching video details in chunks (this can take a bit)...")
    # Fetch metadata in chunks
    id_to_item = {}
    fetched = 0
    for chunk in _chunk([v for _ts, v in ordered], 50):
        try:
            items = youtube.video_details(chunk)
            for it in items:
                if it.get("id"):
                    id_to_item[it["id"]] = it
            fetched += len(chunk)
        except Exception as e:
            print("[warn] videos.list failed for a chunk:", e)
        sleep(0.15)

    print(f"[import] Details fetched for ~{len(id_to_item)} IDs (requested {fetched})")

    # Filter long-form videos
    def _is_longform(v):
        dur = int(v.get("duration_sec", 0) or 0)
        title_low = (v.get("title") or "").lower()
        if dur < MIN_DURATION_SEC:
            return False
        if "#short" in title_low or "#shorts" in title_low or "shorts" in title_low:
            return False
        return True

    long_ids = set(vid for vid, it in id_to_item.items() if _is_longform(it))
    print(f"[import] Long-form candidates (≥{MIN_DURATION_SEC}s): {len(long_ids)}")

    # Pick most recent
    recent_long = [(ts, vid) for ts, vid in ordered if vid in long_ids]
    recent_long = recent_long[-MAX_IMPORT:]
    print(f"[import] Will import {len(recent_long)} most-recent long-form videos (target {MAX_IMPORT})")

    if not recent_long:
        print("[import] No long-form videos met the filters. "
              "Try increasing SCAN_WINDOW or lowering LONG_MIN_SEC.")
        return

    # Upsert metadata
    to_upsert = [id_to_item[vid] for _ts, vid in recent_long if vid in id_to_item]
    store.upsert_videos(to_upsert)

    # Add history
    for ts, vid in recent_long:
        store.add_history(vid, dwell_sec=0, disliked=0, skipped=0, ts_override=ts, log_to_csv=True)

    print(f"[import] Done. Imported {len(recent_long)} videos into DB and CSV.")
    print("CSV:", store.history_csv_path())

def main():
    # CLI entry: validate args and run import_watch_history().
    if len(sys.argv) < 2:
        print("Usage: python -m src.import_takeout /full/path/to/watch-history.json")
        sys.exit(1)
    p = Path(sys.argv[1]).expanduser().resolve()
    if not p.exists():
        print("File not found:", p); sys.exit(1)
    if p.suffix.lower() == ".zip":
        print("Please unzip your Takeout first and pass the JSON path."); sys.exit(2)
    if p.name != "watch-history.json":
        print("[warn] File name isn’t 'watch-history.json' — continuing anyway.")
    import_watch_history(p)

if __name__ == "__main__":
    main()
