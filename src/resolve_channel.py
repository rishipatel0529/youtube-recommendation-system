"""
resolve_channel.py
Helpers for resolving YouTube channel references (IDs, URLs, handles, or names).
"""

import os, re, requests
from urllib.parse import urlparse
from . import store

YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

def _is_channel_id(s: str) -> bool:
    # Check if string is a valid YouTube channel ID
    return bool(re.fullmatch(r"UC[0-9A-Za-z_-]{22}", s or ""))

def _parse_channel_url(s: str):
    # Extract (channel_id, handle_or_name) from a YouTube URL/handle.
    try:
        s = s.strip()
        if s.startswith("@"):
            return None, s
        u = urlparse(s)
        host = (u.netloc or "").lower()
        if "youtube.com" not in host and "youtu.be" not in host:
            return None, None
        parts = [p for p in (u.path or "").split("/") if p]
        if not parts:
            return None, None
        if parts[0].lower() == "channel" and len(parts) >= 2 and _is_channel_id(parts[1]):
            return parts[1], None
        if parts[0].startswith("@"):
            return None, "@" + parts[0].lstrip("@")
        return None, parts[-1]
    except Exception:
        return None, None

def _yt_get(endpoint, **params):
    # Wrapper for YouTube Data API calls with API key + error handling
    key = os.getenv("YT_API_KEY") or os.getenv("YOUTUBE_API_KEY")
    if not key:
        return {}
    params = {**params, "key": key}
    try:
        r = requests.get(f"{YOUTUBE_API}/{endpoint}", params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def _resolve_handle(handle: str):
    # Resolve @handle -> (channel_id, title) using API (with fallback search)
    j = _yt_get("channels", part="snippet", forHandle=handle)
    items = (j.get("items") or [])
    if items:
        it = items[0]
        cid = it.get("id")
        title = (it.get("snippet") or {}).get("title") or ""
        if _is_channel_id(cid):
            return cid, title
    j = _yt_get("search", part="snippet", type="channel", q=handle, maxResults=5)
    for it in j.get("items", []):
        if (it.get("snippet", {}).get("title") or "").lower() == handle.lstrip("@").lower():
            return it["id"]["channelId"], it["snippet"]["title"]
    if j.get("items"):
        it = j["items"][0]
        return it["id"]["channelId"], it["snippet"]["title"]
    return None, None

def resolve_channel_ref(ref: str):
    """
    Resolve user input (id/url/@handle/name) → (channel_id, title, url).
    Returns (None, None, None) if resolution fails.
    """
    if not ref:
        return None, None, None
    ref = ref.strip()

    if _is_channel_id(ref):
        cid = ref
        return cid, None, f"https://www.youtube.com/channel/{cid}"

    cid, handle_or_name = _parse_channel_url(ref)
    if cid:
        return cid, None, f"https://www.youtube.com/channel/{cid}"
    if handle_or_name and handle_or_name.startswith("@"):
        cid, title = _resolve_handle(handle_or_name)
        if cid:
            return cid, title, f"https://www.youtube.com/channel/{cid}"

    with store.get_conn() as conn:
        rows = conn.execute("""
            SELECT channel_id, MAX(channel_title) AS title, COUNT(*) AS n
            FROM videos
            WHERE LOWER(COALESCE(channel_title,'')) = LOWER(?)
            GROUP BY channel_id
            ORDER BY n DESC
            LIMIT 2
        """, (ref,)).fetchall()
    if len(rows) == 1 and _is_channel_id(rows[0][0]):
        cid, title = rows[0][0], rows[0][1] or ref
        return cid, title, f"https://www.youtube.com/channel/{cid}"

    j = _yt_get("search", part="snippet", type="channel", q=ref, maxResults=5)
    items = j.get("items") or []
    for it in items:
        title = it["snippet"]["title"]
        if title.lower() == ref.lower():
            cid = it["id"]["channelId"]
            return cid, title, f"https://www.youtube.com/channel/{cid}"
    if items:
        it = items[0]
        cid = it["id"]["channelId"]
        title = it["snippet"]["title"]
        return cid, title, f"https://www.youtube.com/channel/{cid}"

    return None, None, None
