# src/expand_from_history.py
import re
from collections import Counter, deque
from pathlib import Path
from .utils import load_env

from . import store, youtube

# How many recent items to consider from history (DB ∪ CSV)
HIST_WINDOW = 400
# How many to pull per source
PER_CHANNEL   = 15
PER_RELATED   = 15
PER_KEYWORDS  = 15

def _kw_from_title(t, k=6):
    words = re.findall(r"[A-Za-z0-9]{3,}", (t or "").lower())
    return " ".join(words[:k]) if words else ""

def _read_csv_history(max_rows=2000):
    """Return list of (video_id, ts) most-recent-first from CSV."""
    p = Path(store.history_csv_path())
    out = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        # skip header
        next(f, None)
        for line in f:
            # ts,video_id,title,channel_title,dwell_sec,disliked,skipped
            parts = line.rstrip("\n").split(",", 3)
            if len(parts) < 2: 
                continue
            ts = parts[0]
            vid = parts[1]
            try:
                ts = int(ts)
            except Exception:
                continue
            out.append((vid, ts))
    # CSV is chronological append, so last lines are newest; reverse to newest-first
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:max_rows]

def _read_db_history(max_rows=2000):
    """Return list of (video_id, ts) most-recent-first from DB."""
    rows = store.get_history(n=max_rows)  # [(video_id, ts, dwell, disliked, skipped)]
    out = [(vid, ts) for (vid, ts, *_rest) in rows]
    # get_history is DESC already, but let’s be safe
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def _dedup_preserve_order(pairs):
    seen = set()
    deduped = []
    for vid, ts in pairs:
        if vid in seen:
            continue
        seen.add(vid)
        deduped.append((vid, ts))
    return deduped

def run():
    load_env()
    store.init_db()

    # --- before stats
    all_before = store.fetch_all_videos()
    all_ids_before = {v["id"] for v in all_before}
    print(f"[expand] before: videos in cache = {len(all_before)}")

    # 1) Build a recent history window from DB ∪ CSV
    db_hist  = _read_db_history(max_rows=2000)
    csv_hist = _read_csv_history(max_rows=2000)
    combined = _dedup_preserve_order(sorted(db_hist + csv_hist, key=lambda x: x[1], reverse=True))
    if not combined:
        print("[expand] No history found in DB or CSV. Nothing to expand.")
        return
    recent = combined[:HIST_WINDOW]

    # 2) Make sure the videos table has details for these history IDs
    missing = [vid for (vid, _ts) in recent if vid not in all_ids_before]
    if missing:
        print(f"[expand] fetching details for {len(missing)} recent history IDs not in cache...")
        for i in range(0, len(missing), 50):
            chunk = missing[i:i+50]
            try:
                items = youtube.video_details(chunk)  # long-form filtering happens inside
                store.upsert_videos(items)
            except Exception as e:
                print("[expand] details chunk failed:", e)

    # Refresh after upserts
    vids_map = {v["id"]: v for v in store.fetch_all_videos()}

    # Extract top channels from the *enriched* recent history
    ch_counts = Counter()
    titles_q  = deque()
    rel_seeds = deque()
    for vid, _ts in recent:
        v = vids_map.get(vid)
        if not v:
            continue
        if v.get("channel_id"):
            ch_counts[v["channel_id"]] += 1
        if v.get("title"):
            titles_q.append(v["title"])
        rel_seeds.append(vid)

    top_channels = [ch for ch,_ in ch_counts.most_common(12)]
    print(f"[expand] top channels from history: {len(top_channels)}")

    # 3) Expand: channel uploads
    upserted = 0
    for ch in top_channels:
        try:
            items = youtube.channel_uploads_videos(ch, max_results=PER_CHANNEL)
            store.upsert_videos(items)
            upserted += len(items)
        except Exception as e:
            print("[expand] channel uploads failed:", e)

    # 4) Expand: related videos for last N history watches (dedup by order)
    for vid in list(rel_seeds)[:60]:
        try:
            items = youtube.related(vid, max_results=PER_RELATED)
            store.upsert_videos(items)
            upserted += len(items)
        except Exception as e:
            print("[expand] related failed:", e)

    # 5) Expand: keyword search from recent titles
    seen_q = set()
    for t in list(titles_q)[:80]:
        q = _kw_from_title(t)
        if not q or q in seen_q:
            continue
        seen_q.add(q)
        try:
            items = youtube.search_by_keywords(q, max_results=PER_KEYWORDS)
            store.upsert_videos(items)
            upserted += len(items)
        except Exception as e:
            print("[expand] keyword search failed:", e)

    # --- after stats
    all_after = store.fetch_all_videos()
    print(f"[expand] after: videos in cache = {len(all_after)}  (+{len(all_after) - len(all_before)})")
    print(f"[expand] upserted candidates (raw count across calls) ~ {upserted}")
    print("[expand] Done.")
    
if __name__ == "__main__":
    run()
