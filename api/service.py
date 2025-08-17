"""
service.py
Business logic layer between FastAPI routes and core engine (src/*).
Handles recommendations, search, history, interactions, AB reports,
and actions like block/save/subscribe with persistence in SQLite.
"""

import os, time, uuid, sqlite3, re
from typing import List, Dict, Any, Optional, Tuple
from src.main import build_page, set_search_query, set_trending, set_category
from src import store, categories
from src.resolve_channel import resolve_channel_ref
from src import youtube
from collections import defaultdict

try:
    from . import service as _self
except Exception:
    _self = None

def clear_pagecache():
    # Best-effort clear of in-process page cache (no-op if absent).
    try:
        from .service import _PAGECACHE # type: ignore[attr-defined]
        _PAGECACHE.clear()
    except Exception:
        pass

def get_conn() -> sqlite3.Connection:
    # Return a sqlite3 connection from the store module.
    return store.get_conn()

SESSION_ID = os.getenv("SESSION_ID", str(uuid.uuid4()))
LONG_MIN_SEC = int(os.getenv("LONG_MIN_SEC", "180"))

def _as_list(x):
    # Split a delimited string into a list; returns [] if not list/str.
    if isinstance(x, list): return x
    if isinstance(x, str):
        parts = [p.strip() for p in re.split(r"[|,;]+", x) if p.strip()]
        return parts
    return []

# Pager entrypoints used by the API

def recommend_via_pager(user_id: str, page: int, page_size: int, variant: str) -> List[Dict]:
    # Build a recommendation page (clears search mode).
    set_search_query("")
    items = build_page(page=page, page_size=page_size, variant=variant)
    return _normalize_items(items, variant, page, page_size, source="recommend")

def search_via_pager(q: str, page: int, page_size: int) -> List[Dict]:
    # Build a search results page (fixed variant 'A').
    set_search_query(q or "")
    items = build_page(page=page, page_size=page_size, variant="A")
    return _normalize_items(items, "A", page, page_size, source="search")

def _normalize_items(items: List[Dict], variant: str, page: int, page_size: int, source: str = "recommend") -> List[Dict]:
    # Map internal dicts to the public RecItem shape with correct ranks.
    out = []
    for i, it in enumerate(items[:page_size], start=1):
        out.append({
            "video_id": it.get("id","") or it.get("video_id",""),
            "title": it.get("title",""),
            "channel": it.get("channel_title","") or it.get("channel",""),
            "duration_sec": int(it.get("duration_sec") or 0),
            "why": it.get("why") or [],
            "rank": i + (page-1)*page_size,
            "variant": variant,
            "channel_id": it.get("channel_id")
        })
    return out

def record_history_for_event(event: str, video_id: str, dwell_sec: float | None = None):
    # Persist watch/like/skip history; errors are swallowed to keep UX fast.
    if not video_id or not event:
        return
    try:
        if event == "like":
            store.add_history(video_id, dwell_sec=int(dwell_sec or 0), disliked=0, skipped=0, log_to_csv=True)
        elif event == "dislike":
            store.add_history(video_id, dwell_sec=int(dwell_sec or 0), disliked=1, skipped=0, log_to_csv=True)
        elif event == "skip":
            store.add_history(video_id, dwell_sec=int(dwell_sec or 0), disliked=1, skipped=1, log_to_csv=False)
    except Exception:
        pass

# Legacy/internal helpers (optional)

def recommend(user_id: str, page: int, page_size: int, variant: str) -> List[Dict]:
    # Legacy recommend entrypoint; use recommend_via_pager where possible.
    items = build_page(page=page, page_size=page_size, variant=variant)
    return _normalize_items(items, variant, page, page_size, source="recommend")

def search(q: str, page: int, page_size: int, variant: str):
    # Direct YouTube search; usually prefer search_via_pager().
    q = (q or "").strip()
    vids = []
    try:
        cid, _title, _url = resolve_channel_ref(q)
    except Exception:
        cid = None

    try:
        if cid:
            vids = youtube.channel_uploads_videos(cid, max_results=page*page_size*2)
        else:
            vids = youtube.search_by_keywords(q, max_results=page*page_size*2)
    except Exception:
        vids = []

    return _normalize_items(vids[(page-1)*page_size:], variant, page, page_size)

# Metrics & persistence

def log_interaction(payload: Dict):
    # Append a single interaction event to sqlite (interaction_log table).
    conn = get_conn()
    with conn:
        conn.execute(
            "INSERT INTO interaction_log(ts,session_id,user_id,video_id,rank,variant,event,dwell_sec,page,is_search,query) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                int(time.time()),
                SESSION_ID,
                payload.get("user_id","local"),
                payload["video_id"],
                payload.get("rank"),
                payload.get("variant"),
                payload.get("event"),
                payload.get("dwell_sec"),
                payload.get("page"),
                int(payload.get("is_search") or 0),
                payload.get("query"),
            ),
        )

def ab_report_since(since_ts: int) -> Dict:
    # Aggregate per-variant counts and compute simple CTR/like_rate since a timestamp.
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""
    SELECT variant, event, COUNT(*) 
    FROM interaction_log 
    WHERE ts >= ? 
    GROUP BY variant, event
    """, (since_ts,))
    by = {}
    for v,e,c in cur.fetchall():
        by.setdefault(v,{})[e]=c
    def rate(d, num, den):
        n = d.get(num,0); m = d.get(den,0)
        return float(n)/m if m else 0.0
    out = {}
    for v, d in by.items():
        out[v] = {"events": d, "ctr": rate(d,"watch","impression"), "like_rate": rate(d,"like","watch")}
    keys = list(out.keys())
    delta = {}
    if len(keys)>=2:
        a,b = keys[0], keys[1]
        delta = {"ctr": out.get(b,{}).get("ctr",0)-out.get(a,{}).get("ctr",0),
                 "like_rate": out.get(b,{}).get("like_rate",0)-out.get(a,{}).get("like_rate",0)}
    return {"by_variant": out, "delta": delta}

# Actions (block/save/subscriptions/categories)

def block_channel(channel_id: str, channel_title: Optional[str] = None):
    # Block a channel and clear page cache.
    if not channel_id: return {"ok": False}
    if not channel_title:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(NULLIF(channel_title,''),?) FROM videos WHERE channel_id=? ORDER BY rowid DESC LIMIT 1",
                (channel_id, channel_id)
            ).fetchone()
            channel_title = row[0] if row else channel_id
    store.block_channel(channel_id, channel_title or channel_id)
    clear_pagecache()
    return {"ok": True}

def save_video(video_id: str):
    # Save a video to favorites, inferring channel metadata from DB.
    if not video_id: return {"ok": False}
    with get_conn() as conn:
        row = conn.execute("SELECT channel_id, channel_title FROM videos WHERE id=? LIMIT 1", (video_id,)).fetchone()
    if row:
        store.add_favorite_video(video_id, row[0] or "", row[1] or "")
        return {"ok": True}
    return {"ok": False}

def save_channel(channel_id: str, channel_title: Optional[str] = None):
    # Save a channel to favorites (best-effort title lookup).
    if not channel_id: return {"ok": False}
    if not channel_title:
        with get_conn() as conn:
            row = conn.execute("SELECT channel_title FROM videos WHERE channel_id=? ORDER BY rowid DESC LIMIT 1", (channel_id,)).fetchone()
        channel_title = (row[0] if row and row[0] else channel_id)
    store.add_favorite_channel(channel_id, channel_title)
    return {"ok": True}

def subscribe(channel_id: str, channel_title: Optional[str] = None, channel_url: Optional[str] = None):
    # Create a local subscription record.
    if not channel_id: return {"ok": False}
    store.add_subscription(channel_id, channel_title, channel_url)
    return {"ok": True}

def unsubscribe(channel_id: str):
    # Remove a local subscription record.
    if not channel_id: return {"ok": False}
    store.remove_subscription(channel_id)
    return {"ok": True}

def list_subscriptions():
    # Return all local subscription rows.
    return store.get_subscriptions()

def list_categories(limit: int = 50):
    # Return top categories and counts for filtering.
    rows = store.get_top_categories(limit=limit)
    return [{"id": cid, "name": categories.label(cid), "count": n} for cid, n in rows]

def set_trending_only(on: bool):
    # Toggle trending-only mode and clear cache.
    set_trending(bool(on))
    clear_pagecache()
    return {"ok": True}

def set_active_category(category_id: Optional[str]):
    # Set or clear active category and clear cache.
    set_category(category_id)
    clear_pagecache()
    return {"ok": True}
