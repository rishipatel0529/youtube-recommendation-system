import os, re, requests
from urllib.parse import urlparse
from . import store

YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

def _is_channel_id(s: str) -> bool:
    return bool(re.fullmatch(r"UC[0-9A-Za-z_-]{22}", s or ""))

def _parse_channel_url(s: str):
    """Return (channel_id, handle_or_name) from a YouTube URL/handle; both may be None."""
    try:
        s = s.strip()
        if s.startswith("@"):  # handle only
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
        # /@handle form
        if parts[0].startswith("@"):
            return None, "@" + parts[0].lstrip("@")
        # legacy /c/<custom> or other vanity → treat as name
        return None, parts[-1]
    except Exception:
        return None, None

def _yt_get(endpoint, **params):
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
    # channels?forHandle=@handle (preferred). Fallback via search if needed.
    j = _yt_get("channels", part="snippet", forHandle=handle)
    items = (j.get("items") or [])
    if items:
        it = items[0]
        cid = it.get("id")
        title = (it.get("snippet") or {}).get("title") or ""
        if _is_channel_id(cid):
            return cid, title
    # fallback: search
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
    Resolve a user-supplied reference (id/url/@handle/name) -> (channel_id, title, url)
    Returns (None, None, None) on failure.
    """
    if not ref:
        return None, None, None
    ref = ref.strip()

    # 1) direct channel id
    if _is_channel_id(ref):
        cid = ref
        return cid, None, f"https://www.youtube.com/channel/{cid}"

    # 2) URL or @handle
    cid, handle_or_name = _parse_channel_url(ref)
    if cid:
        return cid, None, f"https://www.youtube.com/channel/{cid}"
    if handle_or_name and handle_or_name.startswith("@"):
        cid, title = _resolve_handle(handle_or_name)
        if cid:
            return cid, title, f"https://www.youtube.com/channel/{cid}"

    # 3) exact title from local DB (unique)
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

    # 4) YouTube API channel search
    j = _yt_get("search", part="snippet", type="channel", q=ref, maxResults=5)
    items = j.get("items") or []
    # prefer exact case-insensitive title match
    for it in items:
        title = it["snippet"]["title"]
        if title.lower() == ref.lower():
            cid = it["id"]["channelId"]
            return cid, title, f"https://www.youtube.com/channel/{cid}"
    # else take the top result
    if items:
        it = items[0]
        cid = it["id"]["channelId"]
        title = it["snippet"]["title"]
        return cid, title, f"https://www.youtube.com/channel/{cid}"

    return None, None, None
