"""
youtube.py
YouTube Data API utilities:
- Fetch video details, popular/trending, related, and channel uploads
- Smart search (morphology, channel detection, dedupe)
- Simple JSON cache with TTL for reducing API quota usage
"""

import os, requests, json, re, time, random, pathlib, difflib, itertools
from typing import List, Dict

BASE = "https://www.googleapis.com/youtube/v3/"

# Config & cache paths
MIN_LONG_SEC = int(os.getenv("LONG_MIN_SEC", "180")) # minimum duration for "long-form" videos
CACHE_TTL_HOURS = int(os.getenv("YT_CACHE_TTL_HOURS", "6"))
DATA_DIR = pathlib.Path(os.getenv("YT_REC_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = DATA_DIR / "yt_video_cache.json"

# Disk cache helpers

def _load_cache():
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(cache):
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass

def _cache_get(ids):
    # Return (fresh_entries, stale_ids, full_cache).
    cache = _load_cache()
    now = time.time()
    fresh = {}
    remaining = []
    ttl = CACHE_TTL_HOURS * 3600
    for vid in ids:
        ent = cache.get(vid)
        if ent and (now - ent.get("ts", 0) <= ttl):
            fresh[vid] = ent.get("data", {})
        else:
            remaining.append(vid)
    return fresh, remaining, cache

def _cache_set(cache, items):
    # Update cache with fresh items and save to disk.
    now = time.time()
    for it in items:
        vid = it.get("id")
        if not vid: 
            continue
        cache[vid] = {"ts": now, "data": it}
    _save_cache(cache)

# API request helper

def _get(url, params, retries=3):
    # GET request with retries/backoff; raises if exhausted.
    key = os.getenv("YT_API_KEY", "").strip()
    if not key:
        raise SystemExit("YT_API_KEY is empty. Set it in .env or the environment.")
    p = dict(params or {})
    p["key"] = key
    backoff = 1.0
    for attempt in range(retries):
        r = requests.get(BASE + url, params=p, timeout=20)
        if r.status_code < 400:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff = min(backoff * 2, 8.0)
            continue
        try:
            err = r.json()
        except Exception:
            err = {"raw": r.text}
        raise requests.HTTPError(
            f"{r.status_code} {r.reason} on {url} with params {p} :: {json.dumps(err)[:600]}",
            response=r
        )
    r.raise_for_status()

# Utilities

def iso8601_duration_to_seconds(s):
    # Convert ISO-8601 duration (e.g. PT5M30S) -> seconds.
    if not s:
        return 0
    s = s.upper()
    h = m = sec = 0
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        elif ch == "H":
            h = int(num or 0); num = ""
        elif ch == "M":
            m = int(num or 0); num = ""
        elif ch == "S":
            sec = int(num or 0); num = ""
    return h*3600 + m*60 + sec

# Basic video fetches

def most_popular(max_results=50):
    # Fetch region's trending/most popular videos.
    region = os.getenv("REGION", "US")
    data = _get("videos", {
        "chart": "mostPopular",
        "part": "snippet,statistics,contentDetails",
        "regionCode": region,
        "maxResults": min(max_results, 50)
    })
    return parse_videos(data)

def video_snippet(video_id):
    # Return (snippet dict, duration_seconds) for one video.
    det = _get("videos", {"id": video_id, "part": "snippet,contentDetails"})
    items = det.get("items", [])
    if not items:
        return {}, 0
    sn = items[0].get("snippet", {}) or {}
    cd = items[0].get("contentDetails", {}) or {}
    dur = iso8601_duration_to_seconds(cd.get("duration"))
    return sn, dur

def keyword_query_from_snippet(sn):
    # Build lightweight keyword query from a snippet's title/tags.
    title = sn.get("title", "") or ""
    tags = sn.get("tags", []) or []
    words = re.findall(r"[A-Za-z0-9]{3,}", title.lower())
    kws = (tags[:6] if tags else words[:6])
    if not kws:
        kws = words[:5]
    return " ".join(kws)

def search_by_keywords(q, max_results=25, relevance_language="en"):
    # Search YouTube by free-text query.
    srch = _get("search", {
        "part": "snippet",
        "type": "video",
        "q": q,
        "maxResults": min(max_results, 50),
        "relevanceLanguage": relevance_language
    })
    ids = [i["id"]["videoId"] for i in srch.get("items", []) if i.get("id", {}).get("videoId")]
    return video_details(ids)

def channel_uploads_videos(channel_id, max_results=25):
    # Fetch recent uploads for a channel.
    ch = _get("channels", {"id": channel_id, "part": "contentDetails"})
    items = ch.get("items", [])
    if not items:
        return []
    up = items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    if not up:
        return []
    pl = _get("playlistItems", {
        "part": "snippet,contentDetails",
        "playlistId": up,
        "maxResults": min(max_results, 50)
    })
    ids = [it.get("contentDetails", {}).get("videoId") for it in pl.get("items", []) if it.get("contentDetails", {}).get("videoId")]
    ids = [i for i in ids if i]
    return video_details(ids)

def video_details(ids):
    # Hydrate video IDs into full details (uses cache).
    if not ids:
        return []
    fresh, remaining, cache = _cache_get(ids)
    out = []
    if fresh:
        out.extend(_items_to_normalized(list(fresh.values())))
    for j in range(0, len(remaining), 50):
        chunk = remaining[j:j+50]
        det = _get("videos", {"id": ",".join(chunk), "part": "snippet,statistics,contentDetails"})
        items = det.get("items", [])
        _cache_set(cache, items)
        out.extend(_items_to_normalized(items))
    return out

def related(video_id, max_results=25):
    # Fetch videos related to a given video ID.
    try:
        data = _get("search", {"part": "snippet", "type": "video", "relatedToVideoId": video_id, "maxResults": min(max_results, 50)})
        ids = [i["id"]["videoId"] for i in data.get("items", []) if i.get("id", {}).get("videoId")]
        if ids:
            return video_details(ids)
    except Exception:
        return []
    return []

# Smart search (morphology + channel detection)
_SUFFIXES = ["", "s", "es", "er", "ers", "or", "ors", "ed", "ing", "ion", "ions", "ial", "al"]

def _stemish(token: str) -> str:
    # Crude stemming: strip common suffixes if length >= 3.
    for suf in ["ing", "ions", "ion", "ers", "er", "ors", "or", "es", "s", "ed", "al", "ial"]:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token

def _expand_query_terms(q: str, max_variants: int = 6) -> List[str]:
    # Expand query into morphological variants (for recall).
    tokens = re.findall(r"[A-Za-z0-9']+", (q or "").lower())
    if not tokens:
        return [q]

    per_token_variants: List[List[str]] = []
    for t in tokens:
        base = _stemish(t)
        variants = {t, base}
        for suf in _SUFFIXES:
            variants.add(base + suf)
        keep = sorted(variants, key=lambda s: (len(s), s))[:4]
        per_token_variants.append(keep)

    combos = []
    for combo in itertools.product(*per_token_variants):
        combos.append(" ".join(combo))
        if len(combos) >= max_variants:
            break

    out = [q] + [c for c in combos if c != q]
    seen = set(); deduped = []
    for s in out:
        if s not in seen:
            seen.add(s); deduped.append(s)
    return deduped

def search_channels_basic(q: str, max_results: int = 20) -> List[Dict]:
    # Lightweight channel search (id + title only).
    data = _get("search", {
        "part": "snippet",
        "type": "channel",
        "q": q,
        "maxResults": min(max_results, 50)
    })
    out = []
    for it in data.get("items", []):
        ch_id = (it.get("id") or {}).get("channelId")
        sn = it.get("snippet", {}) or {}
        if ch_id:
            out.append({"channel_id": ch_id, "channel_title": sn.get("title", "")})
    return out


def _best_channel_hits(query: str, candidates: List[Dict], topk: int = 3, min_ratio: float = 0.60) -> List[Dict]:
    # Pick closest channel name matches to query (fuzzy).
    scored = []
    ql = (query or "").lower().strip()
    for c in candidates:
        name = (c.get("channel_title") or "").strip()
        if not name:
            continue
        ratio = difflib.SequenceMatcher(None, ql, name.lower()).ratio()
        scored.append((ratio, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for r, c in scored if r >= min_ratio][:topk]


def _uploads_video_ids(channel_id: str, max_results: int = 50) -> List[str]:
    # Fetch recent upload IDs for a channel (quota-light).
    ch = _get("channels", {"id": channel_id, "part": "contentDetails"})
    items = ch.get("items", [])
    if not items:
        return []
    uploads_pl = items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    if not uploads_pl:
        return []
    pl = _get("playlistItems", {
        "part": "contentDetails",
        "playlistId": uploads_pl,
        "maxResults": min(max_results, 50)
    })
    ids = [it.get("contentDetails", {}).get("videoId") for it in pl.get("items", []) if it.get("contentDetails", {}).get("videoId")]
    return [i for i in ids if i]

def search_smart(query: str, max_results: int = 60) -> List[Dict]:
    # Smart search: expands query, detects channels, dedupes IDs, hydrates.
    if not query:
        return []

    ids: List[str] = []
    # (1) morphological variants
    variants = _expand_query_terms(query, max_variants=6)
    per = max(5, max_results // max(1, len(variants)))
    for v in variants:
        try:
            srch = _get("search", {
                "part": "snippet",
                "type": "video",
                "q": v,
                "maxResults": min(per, 50)
            })
            ids.extend([i["id"]["videoId"] for i in srch.get("items", []) if i.get("id", {}).get("videoId")])
        except Exception:
            pass
    
    # (2) channel flow
    try:
        ch_raw = search_channels_basic(query, max_results=20)
        best = _best_channel_hits(query, ch_raw, topk=3, min_ratio=0.60)
        for ch in best:
            try:
                up_ids = _uploads_video_ids(ch["channel_id"], max_results=max(10, max_results // 2))
                ids.extend(up_ids)
            except Exception:
                pass
    except Exception:
        pass

    # (3) dedup + hydrate
    seen = set(); uniq = []
    for vid in ids:
        if vid and vid not in seen:
            seen.add(vid); uniq.append(vid)

    out = video_details(uniq)

    # (4) fallback
    if not out:
        try:
            out = search_by_keywords(query, max_results=max_results)
        except Exception:
            out = []

    return out

# Parsing helpers
def _items_to_normalized(items):
    # Wrap list of items into parse_videos().
    data = {"items": items}
    return parse_videos(data)

def parse_videos(data):
    # Normalize API items into flat dicts, filtering out Shorts.
    out = []
    for it in data.get("items", []):
        vid = it.get("id")
        if isinstance(vid, dict):
            vid = vid.get("videoId")
        sn = it.get("snippet", {}) or {}
        st = it.get("statistics", {}) or {}
        cd = it.get("contentDetails", {}) or {}
        dur = iso8601_duration_to_seconds(cd.get("duration"))
        tags = sn.get("tags") or []

        # filter: long-form only, exclude Shorts
        if dur and dur < MIN_LONG_SEC:
            continue

        title = sn.get("title") or ""
        tlow = title.lower()
        if "#short" in tlow or "#shorts" in tlow or "shorts" in tlow:
            continue

        out.append({
            "id": vid,
            "title": title,
            "description": sn.get("description") or "",
            "channel_id": sn.get("channelId") or "",
            "channel_title": sn.get("channelTitle") or "",
            "tags": tags,
            "category_id": sn.get("categoryId") or "",
            "view_count": int(st.get("viewCount", 0)) if st else 0,
            "like_count": int(st.get("likeCount", 0)) if st and st.get("likeCount") else 0,
            "published_at": sn.get("publishedAt") or "",
            "duration_sec": int(dur or 0),
        })

    return out
