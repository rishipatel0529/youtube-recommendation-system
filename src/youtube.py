import os, requests, json, re, time, random, pathlib

BASE = "https://www.googleapis.com/youtube/v3/"

# --- Long-form filter (configurable) ---
MIN_LONG_SEC = int(os.getenv("LONG_MIN_SEC", "180"))  # e.g., 180/240/300

# --- Simple disk cache for videos().list (by id) ---
CACHE_TTL_HOURS = int(os.getenv("YT_CACHE_TTL_HOURS", "6"))
DATA_DIR = pathlib.Path(os.getenv("YT_REC_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = DATA_DIR / "yt_video_cache.json"

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
    now = time.time()
    for it in items:
        vid = it.get("id")
        if not vid: 
            continue
        cache[vid] = {"ts": now, "data": it}
    _save_cache(cache)

def _get(url, params, retries=3):
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
        # Retry on 429 or 5xx
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
    # final attempt failed
    r.raise_for_status()

def iso8601_duration_to_seconds(s):
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

def most_popular(max_results=50):
    region = os.getenv("REGION", "US")
    data = _get("videos", {
        "chart": "mostPopular",
        "part": "snippet,statistics,contentDetails",
        "regionCode": region,
        "maxResults": min(max_results, 50)
    })
    return parse_videos(data)

# ---------- Helpers exposed for main.py to build mixed fallbacks ----------

def video_snippet(video_id):
    """Return (snippet dict, duration_seconds)."""
    det = _get("videos", {"id": video_id, "part": "snippet,contentDetails"})
    items = det.get("items", [])
    if not items:
        return {}, 0
    sn = items[0].get("snippet", {}) or {}
    cd = items[0].get("contentDetails", {}) or {}
    dur = iso8601_duration_to_seconds(cd.get("duration"))
    return sn, dur

def keyword_query_from_snippet(sn):
    title = sn.get("title", "") or ""
    tags = sn.get("tags", []) or []
    words = re.findall(r"[A-Za-z0-9]{3,}", title.lower())
    kws = (tags[:6] if tags else words[:6])
    if not kws:
        kws = words[:5]
    return " ".join(kws)

def search_by_keywords(q, max_results=25, relevance_language="en"):
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
    if not ids:
        return []
    # cache first
    fresh, remaining, cache = _cache_get(ids)
    out = []
    if fresh:
        out.extend(_items_to_normalized(list(fresh.values())))
    # hydrate the rest in batches of 50
    for j in range(0, len(remaining), 50):
        chunk = remaining[j:j+50]
        det = _get("videos", {"id": ",".join(chunk), "part": "snippet,statistics,contentDetails"})
        items = det.get("items", [])
        _cache_set(cache, items)
        out.extend(_items_to_normalized(items))
    return out

def related(video_id, max_results=25):
    try:
        data = _get("search", {"part": "snippet", "type": "video", "relatedToVideoId": video_id, "maxResults": min(max_results, 50)})
        ids = [i["id"]["videoId"] for i in data.get("items", []) if i.get("id", {}).get("videoId")]
        if ids:
            return video_details(ids)
    except Exception:
        return []
    return []

# ---------- parsing ----------

def _items_to_normalized(items):
    data = {"items": items}
    return parse_videos(data)

def parse_videos(data):
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

        # stricter long-form filter
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
