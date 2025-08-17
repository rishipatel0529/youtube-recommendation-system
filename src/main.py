"""
src/main.py
Main entry point for the recommendation system.

- Loads environment config & initializes DB
- Handles recommendation logic (personalized, trending, search, A/B testing)
- Provides an interactive CLI for browsing, blocking, subscribing, etc.
- Exposes API-friendly wrappers for serving recommendations
"""

import sys, time, os, random, re
from tabulate import tabulate
from collections import defaultdict, Counter

from .utils import load_env
from . import store, youtube, recommender, categories, ranker
from .resolve_channel import resolve_channel_ref
from typing import Optional, List, Dict

# knobs
MIN_LONG_SEC = int(os.getenv("LONG_MIN_SEC", "180"))
CHANNEL_CAP  = int(os.getenv("CHANNEL_CAP", "1")) # max per channel per page
BRAND_CAP    = int(os.getenv("BRAND_CAP", "1")) # max per brand per page
RANK_TOP_M   = int(os.getenv("RANK_TOP_M", "60")) # consider top-M ranked before capping
PAGE_MEMORY  = int(os.getenv("PAGE_MEMORY", "10")) # remember last N pages to avoid repeats
EXPLORATION_RATIO = float(os.getenv("EXPLORATION_RATIO", "0.2"))  # % of page from non-subs

# persistent impression cooldown + cross-page channel cap
IMPRESSION_TTL_HOURS = int(os.getenv("IMPRESSION_TTL_HOURS", "48")) # suppress re-show for this many hours
SESSION_CHANNEL_CAP  = int(os.getenv("SESSION_CHANNEL_CAP", "3")) # max per channel across the session

SIMILAR_RATIO = 0.55
SAME_CREATOR_RATIO = 0.15
EXPLORATORY_RATIO = 0.30
FALLBACK_FETCH_TOTAL = 50

TRENDING_FETCH  = os.getenv("TRENDING_FETCH", "1").lower() in {"1","true","yes"}
_trending_only  = False # session toggle

BRAND_STOPS = {"official", "episodes", "full", "channel", "tv", "network", "studios", "production"}

PAGE_SIZE = int(os.getenv("PAGE_SIZE", "10")) # show 10 items per page
RELAX_CAPS_IN_SEARCH = os.getenv("SEARCH_RELAX_CAPS", "1").lower() in {"1","true","yes"}
_search_query = None # active search text (None = off)

# per-session memory of recent pages + new session state
_session_pages = [] # list[set(video_id)]
_session_blocked = set() # set[channel_id]
_category_filter = None # category_id or None
_session_channel_counts = defaultdict(int) # creator quota across the session
_page_counter = 0 # lightweight refresh cadence

# Phase 3: A/B + explainability state
_ab_active = False
_ab_name = None
_ab_page_map = {} # video_id -> "A"/"B" assignment for the current page
_last_contrib = {} # video_id -> feature contributions for trace/why

_search_nextpage_block = set()

API_FAST = os.getenv("API_FAST", "1").lower() in {"1","true","yes"}

try:
    from src.dl.nn_reranker import predict as nn_predict
    USE_NN = bool(int(os.getenv("NN_RERANK", "0")))
except Exception:
    USE_NN = False
    nn_predict = None

# helpers
def _seed_from_local_cache(query: str, max_results=FALLBACK_FETCH_TOTAL):
    # Search the local DB (titles, channel, tags) for query text. No network.
    q = (query or "").strip().lower()
    if not q:
        return []

    toks = [t for t in re.split(r"\W+", q) if t]
    if not toks:
        return []

    vids = store.fetch_all_videos() or []
    scored = []
    for v in vids:
        title   = (v.get("title") or "").lower()
        channel = (v.get("channel_title") or "").lower()
        tags    = " ".join([t.lower() for t in (v.get("tags") or []) if isinstance(t, str)])
        hay     = f"{title} {channel} {tags}"

        # simple token match score
        score = sum(hay.count(t) for t in toks)
        if score <= 0:
            continue

        # keep long-form only
        dur_ok = int(v.get("duration_sec") or 0) >= MIN_LONG_SEC
        if not dur_ok or looks_short_title(v.get("title")):
            continue

        scored.append((score, int(v.get("view_count", 0)), v))

    scored.sort(key=lambda kv: (kv[0], kv[1]), reverse=True)
    return [kv[2] for kv in scored[:max_results]]

def _round_robin_by_channel(ranked, id_to_video, cap, k):
    buckets = defaultdict(list)
    for vid, _ in ranked:
        v = id_to_video.get(vid)
        if not v: continue
        ch = ((v.get("channel_id") or "").strip()
              or (v.get("channel_title") or "").strip().lower())
        buckets[ch].append(vid)

    out_ids, taken_per_ch = [], defaultdict(int)
    while len(out_ids) < k and buckets:
        for ch in list(buckets.keys()):
            if not buckets[ch]:
                del buckets[ch]; continue
            if taken_per_ch[ch] >= cap:
                buckets[ch].clear(); del buckets[ch]; continue
            out_ids.append(buckets[ch].pop(0))
            taken_per_ch[ch] += 1
            if len(out_ids) >= k:
                break
    return out_ids

def looks_short_title(title: str) -> bool:
    t = (title or "").lower()
    return ("#short" in t) or ("#shorts" in t) or (" shorts " in t)

def ensure_seed():
    vids = store.fetch_all_videos()
    if vids:
        return
    items = youtube.most_popular(max_results=50)
    store.upsert_videos(items)

def _duration_bucket(sec):
    s = int(sec or 0)
    if s < 300: return "short"
    if s <= 1200: return "medium"
    return "long"

def brand_key(v):
    title = (v.get("channel_title") or "").lower()
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    toks = [t for t in title.split() if t not in BRAND_STOPS]
    stem = " ".join(toks[:3]).strip()
    return stem or (v.get("channel_id") or "").strip().lower()

_MODEL_CACHE = None
_MODEL_KEY = None

def _make_model_key(all_videos):
    # cheap key: (count, last_id)
    return (len(all_videos), all_videos[-1]["id"] if all_videos else "")

def _get_model(all_videos):
    global _MODEL_CACHE, _MODEL_KEY
    key = _make_model_key(all_videos)
    if _MODEL_CACHE is None or _MODEL_KEY != key:
        _MODEL_CACHE = recommender.build_vectors(all_videos)
        _MODEL_KEY = key
    return _MODEL_CACHE

def expand_candidates_mixed(picked_id):
    rel = youtube.related(picked_id, max_results=FALLBACK_FETCH_TOTAL)
    sn, dur = {}, 0
    try:
        sn, dur = youtube.video_snippet(picked_id)
    except Exception:
        pass
    ch_id = sn.get("channelId", "")
    q = youtube.keyword_query_from_snippet(sn) if sn else ""
    want_bucket = _duration_bucket(dur)

    pool_similar, pool_same, pool_explore = [], [], []
        
    if q:
        try:
            res = youtube.search_smart(q, max_results=FALLBACK_FETCH_TOTAL)
            if ch_id:
                res = [v for v in res if v.get("channel_id") != ch_id]
            if want_bucket != "unknown":
                tmp = [v for v in res if _duration_bucket(v.get("duration_sec", 0)) == want_bucket]
                if tmp:
                    res = tmp
            pool_similar = res
        except Exception:
            pass

    if ch_id:
        try:
            pool_same = youtube.channel_uploads_videos(ch_id, max_results=FALLBACK_FETCH_TOTAL)
        except Exception:
            pass

    try:
        pool_explore = youtube.most_popular(max_results=FALLBACK_FETCH_TOTAL)
    except Exception:
        pass

    if rel:
        rel2 = [v for v in rel if not ch_id or v.get("channel_id") != ch_id]
        pool_similar = (rel2 + pool_similar) if pool_similar else rel2

    def dedup(seq):
        seen = set(); out = []
        for v in seq:
            vid = v.get("id")
            if not vid or vid in seen: continue
            seen.add(vid); out.append(v)
        return out

    pool_similar = dedup(pool_similar); pool_same = dedup(pool_same); pool_explore = dedup(pool_explore)

    target = FALLBACK_FETCH_TOTAL
    n_sim = max(0, int(target * SIMILAR_RATIO))
    n_same = max(0, int(target * SAME_CREATOR_RATIO))
    n_exp  = max(0, int(target * EXPLORATORY_RATIO))

    picked = pool_similar[:n_sim] + pool_same[:n_same] + pool_explore[:n_exp]
    remaining = target - len(picked)
    if remaining > 0:
        backfill = pool_similar[n_sim:] + pool_same[n_same:] + pool_explore[n_exp:]
        picked += backfill[:remaining]

    random.shuffle(picked)
    if picked:
        store.upsert_videos(picked)


def set_search_query(q: str | None):
    # Enable/disable search mode from the API/UI.
    q = (q or "").strip()
    globals()["_search_query"] = q if q else None

# Prefer channel uploads when the query looks like a channel (e.g. "@theb1m", "The B1M", channel URL)
def _seed_from_channel_query(qtxt: str, max_results: int = FALLBACK_FETCH_TOTAL):
    """
    Try to resolve a channel from a user query like '@handle', channel URL, channel id,
    or channel name, and return its recent uploads. Falls back to [] if not resolvable.
    """
    q = (qtxt or "").strip()
    if not q:
        return []

    # resolve_channel_ref returns (channel_id, title, url) for URLs/ids/@handles/names
    try:
        ch_id, _title, _url = resolve_channel_ref(q)
    except Exception:
        ch_id = None

    if not ch_id:
        return []

    try:
        items = youtube.channel_uploads_videos(ch_id, max_results=max_results)
    except Exception:
        items = []

    return items or []

def list_unwatched(limit=200):
    return store.fetch_unwatched(limit=limit)

def history_rows(n=200):
    return store.get_history(n=n)

def show_history():
    print("CSV:", store.history_csv_path())

def _is_allowed_channel(v, blocked_set):
    cid = (v.get("channel_id") or "")
    if cid in blocked_set: return False
    if cid in _session_blocked: return False
    if _category_filter is not None:
        if (v.get("category_id") or "") != _category_filter:
            return False
    return True

# Explainability helpers (session)

def _session_recent_channels():
    # Collect recent channels present across current session pages.
    if not _session_pages: return set()
    ids = set().union(*_session_pages)
    ch = set()
    by_id = {v["id"]: v for v in store.fetch_all_videos()}
    for vid in ids:
        v = by_id.get(vid)
        if v: ch.add(v.get("channel_id",""))
    return ch

def _session_topic_freqs():
    # Rough per-session topic counts from tags/hashtags.
    freq = Counter()
    if not _session_pages: return freq
    ids = set().union(*_session_pages)
    by_id = {v["id"]: v for v in store.fetch_all_videos()}
    for vid in ids:
        v = by_id.get(vid); 
        if not v: continue
        for t in (v.get("tags") or []):
            if isinstance(t, str) and t: freq[t] += 1
    return freq

def _why_not_token(token):
    # Return a one-liner explaining why a given videoId/channelId would be filtered.
    vids = store.fetch_all_videos()
    by_vid = {v["id"]: v for v in vids}
    watched = store.get_all_watched_ids_union_csv()
    blocked = store.get_blocked_channel_ids()

    v = by_vid.get(token)
    if v:
        cid = v.get("channel_id","")
        if v["id"] in watched: return "Filtered: already watched."
        if cid in blocked: return "Filtered: channel is permanently blocked."
        if cid in _session_blocked: return "Filtered: channel is temp-blocked this session."
        if int(v.get("duration_sec") or 0) < MIN_LONG_SEC or looks_short_title(v.get("title")):
            return "Filtered: short-form or below minimum duration."
        if _category_filter and (v.get("category_id") or "") != _category_filter:
            return f"Filtered: not in active category ({categories.label(_category_filter)})."
        return "Not filtered by core rules; likely outranked on this page."
    else:
        if token in blocked: return "Channel filtered: permanently blocked."
        if token in _session_blocked: return "Channel filtered: temp-blocked this session."
        return "Channel not blocked; specific videos may still be outranked."

# UI helpers

def _cmd_help():
    print("""
Keys:
  [1-10]  watch by number   |   s# skip by number
  n       next page (no re-rank; paginate within current set)
  s       search mode (press s, enter keywords; press s again to change)
  !       clear search immediately
  t       toggle trending-only
  c       choose category  |  c clear  clear category
  bt#     temp-block that item’s channel  |  bt manage temp blocks
  b#      PERMANENT block that channel
  f#      save video to favorites  |  fc# save channel  |  F list & remove favorites
  subscribe <#|id|channelId>    subscribe to a channel
  unsubscribe <#|id|channelId>  unsubscribe from a channel
  u       unblock (persistent blocks)
  why #   short reasons     |   trace # feature breakdown
  why-not <videoId|channelId>
  ab start <name>   start team-draft A/B (baseline vs Phase-3 reranker)
  ab stop           stop A/B
  h history CSV path   |   r refresh   |   R reset session   |   q quit
""".strip())

def _print_menu_header():
    print("(u) unblock,  (bt#) temp-block item,  (bt) menu,  (c) category,  (c clear) to clear category,  (f#) save video,  (fc#) save channel,  (F) favorites,")
    print("(n) next page,  (s) search,  (!) clear search,  (t) trending toggle,  (h) history,  (?) help,  (why #),  (trace #),  (why-not <id|channel>),")
    print("(subscribe <#|id|channelId>),  (unsubscribe <#|id|channelId>),  (ab start/stop),  (r) refresh,  (R) reset,  (q) quit.")

def _cmd_unblock_menu():
    rows = store.list_blocked_channels()
    if not rows:
        print("No blocked channels."); return
    for i,(cid, ctitle, _noted) in enumerate(rows, start=1):
        print(f"{i}. {ctitle or cid} [{cid}]")
    s = input("Unblock which number (Enter to cancel): ").strip()
    if not s: return
    try:
        i = int(s)-1
        if 0 <= i < len(rows):
            store.unblock_channel(rows[i][0])
    except Exception:
        pass

def _cmd_category_filter_menu():
    vids = store.fetch_all_videos()
    if not vids:
        print("No videos in DB yet.")
        return

    counts = Counter((v.get("category_id") or "") for v in vids)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], categories.label(kv[0]).lower()))

    def _print_menu(limit=None):
        lst = items if limit is None else items[:limit]
        for idx, (cat_id, cnt) in enumerate(lst, start=1):
            name = categories.label(cat_id); shown_id = cat_id if cat_id else "—"
            print(f"{idx}. {name} ({shown_id}) — {cnt}")
        print("0. Clear")
        if limit is not None and len(items) > limit:
            print("a. Show all")

    limit = 10
    while True:
        _print_menu(limit)
        ch = input("Choose category: ").strip().lower()
        if ch in {"", "q"}: return
        if ch == "a" and len(items) > (limit or 0):
            limit = None; continue
        if ch.isdigit():
            n = int(ch)
            lst = items if limit is None else items[:limit]
            if n == 0:
                globals()["_category_filter"] = None; print("Category filter cleared."); return
            if 1 <= n <= len(lst):
                picked_id = lst[n-1][0]
                globals()["_category_filter"] = picked_id or None
                print(f"Category set: {categories.label(picked_id)}"); return
        print("Invalid choice.")

def _cmd_favorites_menu():
    rows = store.list_favorites()
    if not rows:
        print("No favorites yet."); return
    for i,(vid,cid,ctitle,_noted) in enumerate(rows, start=1):
        tag = "Channel" if vid.startswith("channel::") else "Video"
        print(f"{i}. [{tag}] {ctitle or cid} ({cid}) {vid}")
    s = input("Enter number to remove, or Enter to exit: ").strip()
    if not s: return
    try:
        i = int(s)-1
        vid, cid, _, _ = rows[i]
        key = cid if vid.startswith("channel::") else vid
        store.remove_favorite(key)
    except Exception:
        pass

def _reset_session():
    global _session_pages, _trending_only, _category_filter, _session_blocked, _ab_page_map, _last_contrib
    _session_pages = []; _trending_only = False; _category_filter = None
    _ab_page_map = {}; _last_contrib = {}
    try: _session_blocked.clear()
    except NameError: pass
    try: _session_channel_counts.clear()
    except NameError: pass
    print("Session reset: trending OFF, category cleared, temp blocks cleared, page memory cleared.")

def _status_summary():
    cat_txt = categories.label(_category_filter) if globals().get("_category_filter") else "—"
    tmp_cnt = len(globals().get("_session_blocked", set()) or set())
    try: db_blocked_cnt = len(store.get_blocked_channel_ids())
    except Exception: db_blocked_cnt = 0
    trend_txt = "ON" if globals().get("_trending_only") else "OFF"
    parts = [f"trend={trend_txt}", f"cat={cat_txt}", f"temp_blocked={tmp_cnt}"]
    if db_blocked_cnt: parts.append(f"blocked={db_blocked_cnt}")
    try:
        fav_cnt = len(store.list_favorites())
        if fav_cnt: parts.append(f"favorites={fav_cnt}")
    except Exception:
        pass
    if _ab_active: parts.append(f"AB={_ab_name}")
    return " | ".join(parts)

def _print_status():
    print(f"\n[status] {_status_summary()}\n")

def _manage_temp_blocks():
    global _session_blocked
    if not _session_blocked:
        print("No temp-blocked channels."); return
    title_map = {}
    try:
        for v in store.fetch_all_videos():
            cid = (v.get("channel_id") or "")
            if cid in _session_blocked and cid not in title_map:
                title_map[cid] = v.get("channel_title") or cid
    except Exception:
        pass
    items = sorted([(title_map.get(cid, cid), cid) for cid in _session_blocked], key=lambda x: x[0].lower())
    for i, (title, cid) in enumerate(items, start=1):
        print(f"{i}. {title} [{cid}]")
    ans = input("Unblock which number (Enter to cancel, 0 to unblock ALL): ").strip()
    if ans == "" or ans.lower() == "q": return
    if not ans.isdigit(): print("Cancelled."); return
    j = int(ans)
    if j == 0:
        _session_blocked.clear(); print("Cleared all temporary blocks.")
    elif 1 <= j <= len(items):
        _session_blocked.discard(items[j-1][1]); print("Unblocked (temp).")
    else:
        print("Cancelled.")


def subscribe_cmd(arg: str, current_page_items):
    # Subscribe via # from list, id/url/@handle/name (resolved to a real channel id).
    arg = (arg or "").strip()
    if not arg:
        print("Usage: subscribe <#|id|url|@handle|name>")
        return

    if arg.isdigit():
        i = int(arg) - 1
        if 0 <= i < len(current_page_items):
            ch_id = current_page_items[i]["channel_id"]
            ch_title = current_page_items[i]["channel_title"]
            store.add_subscription(ch_id, ch_title, f"https://www.youtube.com/channel/{ch_id}")
            print(f"Subscribed: {ch_title} [{ch_id}]")
            return

    cid, title, url = resolve_channel_ref(arg)
    if not cid:
        print("Couldn't resolve that channel. Try a channel URL, @handle, channel ID, or pick from the list.")
        return
    store.add_subscription(cid, title, url)
    print(f"Subscribed: {title or cid} [{cid}]")


def unsubscribe_cmd(arg: str, current_page_items):
    # Unsubscribe via # from list, id/url/@handle/name (resolved to a real channel id).
    arg = (arg or "").strip()
    if not arg:
        print("Usage: unsubscribe <#|id|url|@handle|name>")
        return

    if arg.isdigit():
        i = int(arg) - 1
        if 0 <= i < len(current_page_items):
            cid = current_page_items[i]["channel_id"]
            title = current_page_items[i]["channel_title"]
            store.remove_subscription(cid)
            print(f"Unsubscribed: {title} [{cid}]")
            return

    cid, title, _ = resolve_channel_ref(arg)
    if not cid:
        print("Couldn't resolve that channel to unsubscribe.")
        return
    store.remove_subscription(cid)
    print(f"Unsubscribed: {title or cid} [{cid}]")

def pick_interactive(cands):
    _print_status()

    # stamps: [sub] [fresh] [diverse]
    try:
        subs_set = {s["channel_id"] for s in store.get_subscriptions()}
    except Exception:
        subs_set = set()

    seen_ch = set()
    rows = []
    for i, v in enumerate(cands, start=1):
        stamps = []
        if (v.get("channel_id") or "") in subs_set: stamps.append("[sub]")
        # fresh within around 14 days
        try:
            pub = (v.get("published_at") or "").replace("Z","+00:00")
            import datetime as _dt
            dt = _dt.datetime.fromisoformat(pub) if pub else None
            if dt and (_dt.datetime.now(_dt.timezone.utc) - dt).days <= 14:
                stamps.append("[fresh]")
        except Exception:
            pass
        if v.get("channel_id") not in seen_ch:
            stamps.append("[diverse]")
        seen_ch.add(v.get("channel_id"))
        title = v["title"][:80] + (" " + " ".join(stamps) if stamps else "")
        rows.append([i, title, v["channel_title"][:30], v["view_count"]])
    print(tabulate(rows, headers=["#", "Title", "Channel", "Views"]))
    _print_menu_header()

    while True:
        raw = input("Select #: ").strip()
        if not raw:
            print("Invalid input."); continue

        x = raw.lower()

        if raw == "R":
            _reset_session(); return None, None
        if x == "q": sys.exit(0)
        if x == "t":
            global _trending_only
            _trending_only = not _trending_only
            print(f"Trending-only mode: {'ON' if _trending_only else 'OFF'}")
            return None, None
        if raw == "r": return None, None
        if x == "h": show_history(); continue
        if x in {"?","m"}: _cmd_help(); continue

        # pagination
        if x == "n":
            return None, "next"

        # quick clear search
        if x == "!":
            globals()["_search_query"] = None
            print("Search cleared.")
            return None, None

        # search mode (s)
        if x == "s":
            q = input("Enter search keywords (Enter=cancel, '!'=clear): ").strip()
            if q == "":
                print("Search cancelled.")
            elif q == "!":
                globals()["_search_query"] = None
                print("Search cleared.")
            else:
                globals()["_search_query"] = q
                print(f"Search set: {q}")
            return None, None

        # subscribe / unsubscribe
        if x.startswith("subscribe"):
            parts = raw.split(maxsplit=1)
            token = parts[1].strip() if len(parts) > 1 else ""
            if not token:
                print("Usage: subscribe <#|id|url|@handle|name>")
            else:
                subscribe_cmd(token, cands)
            return None, None

        if x.startswith("unsubscribe"):
            parts = raw.split(maxsplit=1)
            token = parts[1].strip() if len(parts) > 1 else ""
            if not token:
                print("Usage: unsubscribe <#|id|url|@handle|name>")
            else:
                unsubscribe_cmd(token, cands)
            return None, None

        # A/B controls
        if x.startswith("ab start"):
            global _ab_active, _ab_name
            parts = raw.split()
            if len(parts) >= 3:
                _ab_active = True
                name = " ".join(parts[2:])
                name = "".join(ch for ch in name if ch.isprintable()).strip()
                import re as _re
                name = _re.sub(r"\s+", " ", name)
                _ab_name = name
                try: store.log_session_event("ab_start", name=_ab_name)
                except Exception: pass
                print(f"A/B started: {_ab_name}")
            else:
                print("Usage: ab start <name>")
            return None, None
        if x == "ab stop":
            if _ab_active:
                try: store.log_session_event("ab_stop", name=_ab_name)
                except Exception: pass
                print("A/B stopped.")
            else:
                print("A/B not active.")
            globals()["_ab_active"] = False; globals()["_ab_name"] = None
            return None, None

        if x == "u":
            _cmd_unblock_menu(); return None, None

        if x.startswith("bt"):
            if x == "bt":
                _manage_temp_blocks(); return None, None
            num = x[2:]
            if not num.isdigit():
                print("Use bt# (e.g., bt3) to temp-block, or 'bt' to manage."); return None, None
            i = int(num)
            if 1 <= i <= len(cands):
                ch_id = cands[i-1].get("channel_id") or ""
                ch_t  = cands[i-1].get("channel_title") or ""
                if ch_id:
                    _session_blocked.add(ch_id); print(f"Temporarily blocked this session: {ch_t or ch_id}")
            else:
                print("Invalid #.")
            return None, None

        if x == "c":
            _cmd_category_filter_menu(); return None, None
        if x == "c clear":
            globals()["_category_filter"] = None; print("Category filter cleared."); return None, None

        if raw == "F" or x == "f":
            _cmd_favorites_menu(); return None, None

        if x.startswith("f") and x[1:].isdigit():
            i = int(x[1:])
            if 1 <= i <= len(cands):
                v = cands[i-1]
                store.add_favorite_video(v.get("id",""), v.get("channel_id",""), v.get("channel_title",""))
                print(f"Saved to favorites: {v.get('title','')[:60]} — {v.get('channel_title','')}")
            return None, None

        if x.startswith("fc") and x[2:].isdigit():
            i = int(x[2:])
            if 1 <= i <= len(cands):
                v = cands[i-1]
                store.add_favorite_channel(v.get("channel_id",""), v.get("channel_title",""))
                print(f"Saved channel to favorites: {v.get('channel_title','')}")
            return None, None

        if x.startswith("b") and x[1:].isdigit():
            i = int(x[1:])
            if 1 <= i <= len(cands):
                ch_id = cands[i-1].get("channel_id") or ""
                ch_t  = cands[i-1].get("channel_title") or ""
                if ch_id:
                    ans = input(f"Block channel permanently: {ch_t or ch_id}? (y/n): ").strip().lower()
                    if ans == "y":
                        store.block_channel(ch_id, ch_t); print(f"Blocked channel: {ch_t or ch_id}")
                    else:
                        print("Cancelled.")
                    return None, None
                else:
                    print("Could not determine channel id for that video."); continue

        if x.startswith("why "):
            num = x.split()[-1]
            if num.isdigit():
                i = int(num)
                if 1 <= i <= len(cands):
                    vid = cands[i-1]["id"]
                    try: subs_set = {s["channel_id"] for s in store.get_subscriptions()}
                    except Exception: subs_set = set()
                    fv = _last_contrib.get(vid, {})
                    reasons = recommender.explain_reasons(
                        {k:v for k,v in fv.items() if isinstance(v,(int,float))},
                        subs_set=subs_set, max_reasons=3
                    )
                    try: store.log_session_event("why", video_id=vid, reasons=reasons)
                    except Exception: pass
                    print("Why:", "; ".join(reasons) if reasons else "No strong signals; likely overall score.")
                else:
                    print("Invalid #.")
            else:
                print("Usage: why <#>")
            continue

        if x.startswith("trace "):
            num = x.split()[-1]
            if num.isdigit():
                i = int(num)
                if 1 <= i <= len(cands):
                    vid = cands[i-1]["id"]
                    fv = _last_contrib.get(vid, {})
                    try: store.log_session_event("trace", video_id=vid)
                    except Exception: pass
                    if fv:
                        parts = [(k, fv[k]) for k in fv.keys() if k != "_total"]
                        parts.sort(key=lambda kv: -abs(kv[1]))
                        print("Score breakdown:")
                        for k,v in parts[:10]:
                            print(f"  {k:16s}  {v:+.4f}")
                        print(f"  {'_total':16s}  {fv.get('_total',0.0):+.4f}")
                    else:
                        print("No feature breakdown available.")
                else:
                    print("Invalid #.")
            else:
                print("Usage: trace <#>")
            continue

        if x.startswith("why-not "):
            token = raw.split(maxsplit=1)[1].split()[-1]
            ans = _why_not_token(token)
            try: store.log_session_event("why_not", token=token, answer=ans)
            except Exception: pass
            print(ans); continue

        if x.startswith("s") and x[1:].isdigit():
            i = int(x[1:])
            if 1 <= i <= len(cands): return cands[i-1], "skip"

        if x.isdigit():
            i = int(x)
            if 1 <= i <= len(cands): return cands[i-1], "watch"

        print("Invalid input.")


def like_feedback():
    v = input("Like this video? (y/n): ").strip().lower()
    return 1 if v == "y" else 0

def _with_jitter(ranked, eps=5e-3):
    return [(vid, score + random.uniform(-eps, eps)) for vid, score in ranked]

def _effective_caps(k):
    """
    Returns (channel_cap, brand_cap, session_channel_cap) for the current context.
    In search mode, we relax caps so you can see multiple items from the same channel.
    """
    if globals().get("_search_query") and RELAX_CAPS_IN_SEARCH:
        # allow up to the page size from a single channel/brand, and loosen the session cap
        return k, k, max(k, SESSION_CHANNEL_CAP)
    return CHANNEL_CAP, BRAND_CAP, SESSION_CHANNEL_CAP


def _light_refresh_if_small():
    if API_FAST:
        return
    global _page_counter
    _page_counter += 1
    if _page_counter % 3 != 0:
        return
    try:
        all_vids = store.fetch_all_videos()
        long_vids = [v for v in all_vids if int(v.get("duration_sec") or 0) >= MIN_LONG_SEC]
        if len(long_vids) < 2000:
            # (1) light trending/popular refresh
            try:
                items = youtube.most_popular(max_results=50)
                if items: store.upsert_videos(items)
            except Exception:
                pass
            # (2) pull uploads from top liked channels (recent history)
            try:
                hist = history_rows(n=2000)
                if hist:
                    by_id = {v["id"]: v for v in all_vids}
                    ch_counts = Counter()
                    for vid_id, *_rest in hist:
                        v = by_id.get(vid_id)
                        if not v: continue
                        if v.get("channel_id"): ch_counts[v["channel_id"]] += 1
                    for cid, _ in ch_counts.most_common(5):
                        try:
                            uploads = youtube.channel_uploads_videos(cid, max_results=30)
                            if uploads: store.upsert_videos(uploads)
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        pass

# Core recommendation
def recommend(k=20):
    _light_refresh_if_small()  # occasionally expand pool when DB is small
    global _session_pages, _ab_page_map, _last_contrib

    # Search seeding (channel-first; then smart; then keywords)
    seed_ids = set()
    seed_pool = []
    if globals().get("_search_query"):
        qtxt = (_search_query or "").strip()

        exclude_once = set(globals().get("_search_nextpage_block") or set())
        globals()["_search_nextpage_block"] = set()

        # (1) channel uploads
        if not API_FAST:
            seed_pool = _seed_from_channel_query(qtxt)

        # (2) smart search
        if not seed_pool and not API_FAST:
            try:
                seed_pool = youtube.search_smart(qtxt, max_results=FALLBACK_FETCH_TOTAL)
            except Exception:
                seed_pool = []

        # (3) keyword search
        if not seed_pool and not API_FAST:
            try:
                seed_pool = youtube.search_by_keywords(qtxt, max_results=FALLBACK_FETCH_TOTAL)
            except Exception:
                seed_pool = []

        if not seed_pool:
            seed_pool = _seed_from_local_cache(qtxt, max_results=FALLBACK_FETCH_TOTAL)

        if seed_pool:
            store.upsert_videos(seed_pool)
            seed_ids = {v.get("id") for v in seed_pool if v.get("id")}

        try:
            store.log_session_event("search_seed", q=qtxt, n=len(seed_ids))
        except Exception:
            pass

    all_videos = store.fetch_all_videos()
    if not all_videos:
        return []

    watched_ids = store.get_all_watched_ids_union_csv()
    blocked = store.get_blocked_channel_ids()

    def is_allowed_channel(v):
        return _is_allowed_channel(v, blocked)

    def is_longform(v):
        return int(v.get("duration_sec") or 0) >= MIN_LONG_SEC and not looks_short_title(v.get("title"))

    unwatched = [
        v for v in all_videos
        if v["id"] not in watched_ids and is_longform(v) and is_allowed_channel(v)
    ]
    if not unwatched:
        return []

    # EFFECTIVE CAPS (relaxed when searching)
    local_ch_cap, local_brand_cap, local_session_cap = _effective_caps(k)

    model = _get_model(all_videos)
    hist = history_rows(n=5000)
    last = hist[0][0] if hist else None
    user_vec = recommender.make_user_profile(model, all_videos, hist) if hist else None

    session_exclude = set().union(*_session_pages) if _session_pages else set()
    if globals().get("_search_query"):
        session_exclude = set()

    recently_shown = set()
    try:
        recently_shown = store.get_recently_impressed_ids(IMPRESSION_TTL_HOURS, 1)
    except Exception:
        recently_shown = set()
    if globals().get("_search_query"):
        recently_shown = set()

    full_exclude = set(watched_ids) | set(session_exclude) | recently_shown

    # Search-first path (long-form only)
    if seed_ids:
        id_to_video = {v["id"]: v for v in store.fetch_all_videos()}

        def _is_longform(v):
            return int(v.get("duration_sec") or 0) >= MIN_LONG_SEC and not looks_short_title(v.get("title"))

        try:
            exclude_once
        except NameError:
            exclude_once = set()

        seed_pool = [
            v for v in (id_to_video.get(vid) for vid in seed_ids if vid in id_to_video)
            if v
            and v["id"] not in watched_ids
            and v["id"] not in exclude_once
            and _is_longform(v)
            and is_allowed_channel(v)
        ]

        if seed_pool:
            # Prefer popular then recent within the seed
            try:
                seed_pool.sort(key=lambda v: (int(v.get("view_count", 0)), (v.get("published_at") or "")), reverse=True)
            except Exception:
                seed_pool.sort(key=lambda v: int(v.get("view_count", 0)), reverse=True)

            out = []
            from collections import defaultdict as _dd
            ch_counts, brand_counts = _dd(int), _dd(int)

            SEARCH_CHANNEL_CAP = PAGE_SIZE
            SEARCH_BRAND_CAP   = PAGE_SIZE

            for v in seed_pool:
                ch = ((v.get("channel_id") or "").strip()
                      or (v.get("channel_title") or "").strip().lower())
                bk = brand_key(v)
                if ch_counts[ch] >= SEARCH_CHANNEL_CAP:  continue
                if brand_counts[bk] >= SEARCH_BRAND_CAP: continue
                out.append(v); ch_counts[ch] += 1; brand_counts[bk] += 1
                if len(out) >= k: break

            globals()["_search_nextpage_block"] = {v["id"] for v in out}

            # no _session_pages append and no log_impressions in search

            try:
                store.log_session_event(
                    "page", k=k, trending=False, category=_category_filter or "",
                    temp_blocked=len(_session_blocked),
                    ab=_ab_name if _ab_active else "",
                    search_q=_search_query
                )
            except Exception:
                pass

            return out[:k]

    # If in SEARCH mode but we failed to seed anything, return empty (no general fallback).
    if globals().get("_search_query") and not seed_ids:
        return []

    # Trending-only path
    if _trending_only:
        try:
            if TRENDING_FETCH and not API_FAST:
                items = youtube.most_popular(max_results=FALLBACK_FETCH_TOTAL)
                if items: store.upsert_videos(items)
        except Exception:
            pass

        popular_pool = sorted(
            [v for v in store.fetch_all_videos()
             if v["id"] not in full_exclude and is_longform(v) and is_allowed_channel(v)],
            key=lambda x: x.get("view_count", 0),
            reverse=True
        )

        out = []
        ch_counts, brand_counts = defaultdict(int), defaultdict(int)
        for v in popular_pool:
            ch = ((v.get("channel_id") or "").strip()
                  or (v.get("channel_title") or "").strip().lower())
            bk = brand_key(v)
            if _session_channel_counts[ch] >= local_session_cap: continue
            if ch_counts[ch] >= local_ch_cap: continue
            if brand_counts[bk] >= local_brand_cap: continue
            out.append(v); ch_counts[ch] += 1; brand_counts[bk] += 1
            _session_channel_counts[ch] += 1
            if len(out) >= k: break

        buffer = list(out)
        if USE_NN and nn_predict and buffer:
            ids = [b["id"] if "id" in b else b.get("video_id") for b in buffer]
            scored = dict(nn_predict([i for i in ids if i]))
            for i, b in enumerate(buffer):
                vid = b.get("id") or b.get("video_id")
                b["_nn_score"] = scored.get(vid, -1e9)
                b["_orig_idx"] = i
            buffer.sort(key=lambda x: (x["_nn_score"], -x["_orig_idx"]), reverse=True)
            out = buffer

        this_page_ids = {v["id"] for v in out}
        if this_page_ids:
            _session_pages.append(this_page_ids)
            if len(_session_pages) > PAGE_MEMORY: _session_pages.pop(0)

        try:
            store.log_impressions([v["id"] for v in out])
        except Exception:
            pass

        try:
            store.log_session_event("page", k=k, trending=True, category=_category_filter or "", temp_blocked=len(_session_blocked))
        except Exception:
            pass
        return out[:k]

    session_recent = _session_topic_freqs()
    session_topics = _session_topic_freqs()

    want_ab = bool(_ab_active and _ab_name)
    if want_ab:
        ranked_A, contrib_A = recommender.rank_hybrid(
            model, user_vec, exclude_ids=full_exclude, k=k*2, last_video=last,
            history_rows=hist, all_videos=all_videos,
            session_recent_channels=session_recent,
            session_topic_freqs=session_topics,
            use_phase3=False, return_contribs=True
        )
        ranked_B, contrib_B = recommender.rank_hybrid(
            model, user_vec, exclude_ids=full_exclude, k=k*2, last_video=last,
            history_rows=hist, all_videos=all_videos,
            session_recent_channels=session_recent,
            session_topic_freqs=session_topics,
            use_phase3=True, return_contribs=True
        )
        idsA = [vid for vid,_ in ranked_A]
        idsB = [vid for vid,_ in ranked_B]
        id_to_video = {v["id"]: v for v in all_videos}

        # In search mode, keep only seeded items
        if seed_ids:
            idsA = [vid for vid in idsA if vid in seed_ids]
            idsB = [vid for vid in idsB if vid in seed_ids]

        from collections import defaultdict as _dd

        def _build_ab_page(idsA, idsB, k):
            out = []
            page_map = {}
            ch_counts, brand_counts = _dd(int), _dd(int)
            ia = ib = 0
            cntA = cntB = 0
            seen_ids = set()

            def _take(owner, ids, idx):
                while idx < len(ids):
                    vid = ids[idx]; idx += 1
                    if vid in seen_ids or vid in session_exclude: continue
                    v = id_to_video.get(vid)
                    if not v: continue
                    if not is_longform(v) or not is_allowed_channel(v): continue
                    ch = ((v.get("channel_id") or "").strip()
                          or (v.get("channel_title") or "").strip().lower())
                    bk = brand_key(v)
                    if _session_channel_counts[ch] >= local_session_cap: continue
                    if ch_counts[ch] >= local_ch_cap: continue
                    if brand_counts[bk] >= local_brand_cap: continue

                    vv = dict(v); vv["ab_variant"] = owner
                    out.append(vv); seen_ids.add(vid)
                    ch_counts[ch] += 1; brand_counts[bk] += 1
                    _session_channel_counts[ch] += 1
                    page_map[vid] = owner
                    return idx, True
                return idx, False

            while len(out) < k and (ia < len(idsA) or ib < len(idsB)):
                owner = 'A' if cntA <= cntB else 'B'
                if owner == 'A':
                    ia, ok = _take('A', idsA, ia)
                    if ok: cntA += 1
                    else:
                        ib, ok2 = _take('B', idsB, ib)
                        if ok2: cntB += 1
                        else: break
                else:
                    ib, ok = _take('B', idsB, ib)
                    if ok: cntB += 1
                    else:
                        ia, ok2 = _take('A', idsA, ia)
                        if ok2: cntA += 1
                        else: break
            return out, page_map

        out, owner = _build_ab_page(idsA, idsB, k)
        _ab_page_map = {v["id"]: v.get("ab_variant","A") for v in out}
        _last_contrib = {**contrib_A, **contrib_B}

        try:
            a_ct = sum(1 for v in out if v.get("ab_variant") == "A")
            b_ct = sum(1 for v in out if v.get("ab_variant") == "B")
            print(f"[ab] page A={a_ct}  B={b_ct}")
        except Exception:
            pass
    else:
        ranked, contrib = recommender.rank_hybrid(
            model, user_vec, exclude_ids=full_exclude, k=k*2, last_video=last,
            history_rows=hist, all_videos=all_videos,
            session_recent_channels=session_recent,
            session_topic_freqs=session_topics,
            use_phase3=True, return_contribs=True
        )
        id_to_video = {v["id"]: v for v in all_videos}
        out, ch_counts, brand_counts = [], defaultdict(int), defaultdict(int)

        try:
            vid_list = [vid for vid, _ in ranked[:k*3]]
            counts = store.get_impression_like_counts(vid_list)
            def _adj(vid, score):
                imp, likes = counts.get(vid, (0,0))
                no_clicks = max(0, imp - likes)
                pen = 0.05 * min(3, no_clicks)
                return score - pen
            ranked = sorted(((vid, _adj(vid, s)) for vid, s in ranked), key=lambda x: x[1], reverse=True)
        except Exception:
            pass

        balanced_ids = _round_robin_by_channel(ranked, id_to_video, local_ch_cap, k*2)
        ranked = [(vid, 0.0) for vid in balanced_ids]

        if seed_ids:
            ranked = [(vid, s) for (vid, s) in ranked if vid in seed_ids]

        for vid, _score in ranked:
            if vid in watched_ids or vid in session_exclude: continue
            v = id_to_video.get(vid)
            if not v or not is_longform(v) or not is_allowed_channel(v): continue
            ch = ((v.get("channel_id") or "").strip() or (v.get("channel_title") or "").strip().lower())
            bk = brand_key(v)
            if _session_channel_counts[ch] >= local_session_cap: continue
            if ch_counts[ch] >= local_ch_cap: continue
            if brand_counts[bk] >= local_brand_cap: continue
            out.append(v); ch_counts[ch]+=1; brand_counts[bk]+=1
            _session_channel_counts[ch] += 1
            if len(out) >= k: break

        _ab_page_map = {}
        _last_contrib = contrib

    USE_SUBS  = os.getenv("USE_SUBS", "0").lower() in {"1","true","yes","2"}
    if len(out) < k:
        target_explore = max(1, int(k * EXPLORATION_RATIO))
        subs_set = {s["channel_id"] for s in store.get_subscriptions()} if USE_SUBS else set()
        explore_pool = [
            v for v in unwatched
            if (v.get("channel_id") or "") not in subs_set
            and is_allowed_channel(v)
            and is_longform(v)
        ]
        if seed_ids:
            explore_pool = [v for v in explore_pool if v["id"] in seed_ids]

        random.shuffle(explore_pool)
        seen_ids = {v["id"] for v in out} | session_exclude
        ch_counts = defaultdict(int, **{((v.get("channel_id") or "").strip()): 1 for v in out})
        brand_counts = defaultdict(int, **{brand_key(v): 1 for v in out})
        added = 0
        for v in explore_pool:
            if v["id"] in seen_ids: continue
            ch = ((v.get("channel_id") or "").strip() or (v.get("channel_title") or "").strip().lower())
            bk = brand_key(v)
            if _session_channel_counts[ch] >= local_session_cap: continue
            if ch_counts[ch] >= local_ch_cap: continue
            if brand_counts[bk] >= local_brand_cap: continue
            out.append(v); ch_counts[ch]+=1; brand_counts[bk]+=1; added += 1
            _session_channel_counts[ch] += 1
            if added >= target_explore or len(out) >= k: break

    if len(out) < k and USE_SUBS:
        subs_set = {s["channel_id"] for s in store.get_subscriptions()}
        sub_pool = [v for v in unwatched if (v.get("channel_id") or "") in subs_set and is_allowed_channel(v)]
        sub_pool = sorted(sub_pool, key=lambda x: x.get("view_count", 0), reverse=True)
        if seed_ids:
            sub_pool = [v for v in sub_pool if v["id"] in seed_ids]

        seen_ids = {v["id"] for v in out} | session_exclude
        ch_counts = defaultdict(int, **{((v.get("channel_id") or "").strip()): 1 for v in out})
        brand_counts = defaultdict(int, **{brand_key(v): 1 for v in out})
        for v in sub_pool:
            if v["id"] in seen_ids: continue
            if not is_longform(v):   continue
            ch = ((v.get("channel_id") or "").strip() or (v.get("channel_title") or "").strip().lower())
            bk = brand_key(v)
            if _session_channel_counts[ch] >= local_session_cap: continue
            if ch_counts[ch] >= local_ch_cap: continue
            if brand_counts[bk] >= local_brand_cap: continue
            out.append(v); ch_counts[ch]+=1; brand_counts[bk]+=1
            _session_channel_counts[ch] += 1
            if len(out) >= k: break

    # General popular backfill
    if len(out) < k:
        popular_pool = sorted(unwatched, key=lambda x: x.get("view_count", 0), reverse=True)
        if seed_ids:
            popular_pool = [v for v in popular_pool if v["id"] in seed_ids]

        seen_ids = {v["id"] for v in out} | session_exclude
        ch_counts = defaultdict(int, **{((v.get("channel_id") or "").strip()): 1 for v in out})
        brand_counts = defaultdict(int, **{brand_key(v): 1 for v in out})
        for v in popular_pool:
            if v["id"] in seen_ids: continue
            if not is_longform(v) or not is_allowed_channel(v): continue
            ch = ((v.get("channel_id") or "").strip() or (v.get("channel_title") or "").strip().lower())
            bk = brand_key(v)
            if _session_channel_counts[ch] >= local_session_cap: continue
            if ch_counts[ch] >= local_ch_cap: continue
            if brand_counts[bk] >= local_brand_cap: continue
            out.append(v); ch_counts[ch]+=1; brand_counts[bk]+=1
            _session_channel_counts[ch] += 1
            if len(out) >= k: break

    buffer = list(out)
    if USE_NN and nn_predict and buffer:
        ids = [b["id"] if "id" in b else b.get("video_id") for b in buffer]
        scored = dict(nn_predict([i for i in ids if i]))
        for i, b in enumerate(buffer):
            vid = b.get("id") or b.get("video_id")
            b["_nn_score"] = scored.get(vid, -1e9)
            b["_orig_idx"] = i
        buffer.sort(key=lambda x: (x["_nn_score"], -x["_orig_idx"]), reverse=True)
        out = buffer

    this_page_ids = {v["id"] for v in out}
    if this_page_ids:
        _session_pages.append(this_page_ids)
        if len(_session_pages) > PAGE_MEMORY: _session_pages.pop(0)

    try:
        store.log_impressions([v["id"] for v in out])
    except Exception:
        pass

    try:
        store.log_session_event("page", k=k, trending=False, category=_category_filter or "", temp_blocked=len(_session_blocked), ab=_ab_name if _ab_active else "")
    except Exception:
        pass
    return out[:k]

# CLI & main

def _print_env_summary():
    key = os.getenv("YT_API_KEY", "")
    key_hint = f"...{key[-4:]}" if key else "(missing)"
    region = os.getenv("REGION", "US")
    use_subs = os.getenv("USE_SUBS", "0")
    cf = os.getenv("USE_CF", "lightfm")
    print(f"[config] REGION={region}  USE_SUBS={use_subs}  USE_CF={cf}  LONG_MIN_SEC={MIN_LONG_SEC}  PAGE_MEMORY={PAGE_MEMORY}")
    if not key:
        print("[config] WARNING: YT_API_KEY missing; API calls will fail.")
    else:
        print(f"[config] YT_API_KEY working.")

def _parse_arg_value(flag):
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
        if a == flag and i + 1 < len(args):
            return args[i + 1]
    return None

def main():
    load_env()
    store.init_db()
    _print_env_summary()

    args = set(sys.argv[1:])
    dry = "--dry-run" in args

    if "--export-blocked" in args:
        if dry:
            n = len(store.list_blocked_channels())
            p = os.path.join(str(store.DATA_DIR), "blocked_channels.csv")
            print(f"[dry-run] Would export {n} blocked channels to: {p}")
            return
        path = store.export_blocked_csv()
        print(f"Blocked channels exported to: {path}")
        return

    if "--export-liked" in args:
        if dry:
            print(f"[dry-run] Would export liked history to: {os.path.join(str(store.DATA_DIR), 'liked_history.csv')}")
            return
        path = store.export_liked_history_csv()
        print(f"Liked history exported to: {path}")
        return

    if "--export-history" in args:
        if dry:
            print(f"[dry-run] Would export full history to: {os.path.join(str(store.DATA_DIR), 'history_full.csv')}")
            return
        # support either name depending on your store.py
        path = None
        try:
            path = store.export_history_csv()
        except Exception:
            path = store.export_full_history_csv()
        print(f"Full history exported to: {path}")
        return

    if "--snapshot-views" in args:
        if dry:
            print("[dry-run] Would snapshot view counts.")
            return
        n, path = store.snapshot_view_counts()
        print(f"Snapshot saved ({n} rows) to: {path}")
        return

    if "--import-subs" in args or any(a.startswith("--import-subs=") for a in sys.argv[1:]):
        csv_path = _parse_arg_value("--import-subs")
        if not csv_path or not os.path.exists(csv_path):
            print("Provide a CSV path: --import-subs path/to/subs.csv"); return
        if dry:
            import csv as _csv
            with open(csv_path, newline="", encoding="utf-8") as f:
                r = _csv.DictReader(f); rows = list(r)
            print(f"[dry-run] Would import {len(rows)} subscriptions from {csv_path}")
            return
        store.import_subscriptions_csv(csv_path)
        print(f"Imported subscriptions from: {csv_path}")
        return

    if "--retrain-ranker" in args:
        save_path = os.path.join(str(store.DATA_DIR), "ranker.pkl")
        metrics = ranker.retrain_from_db(save_path=None if dry else save_path)
        print(f"Ranker retrained. AUC={metrics.get('auc'):.4f}  MAP@10={metrics.get('map10'):.4f}")
        if not dry: print(f"Saved model to: {save_path}")
        return

    if "--partial-fit" in args:
        save_path = os.path.join(str(store.DATA_DIR), "ranker.pkl")
        metrics = ranker.partial_fit_from_db(save_path=None if dry else save_path)
        print(f"Ranker partial-fit. AUC={metrics.get('auc'):.4f}  MAP@10={metrics.get('map10'):.4f}")
        if not dry: print(f"Saved model to: {save_path}")
        return

    if "--ab-report" in args:
        try:
            path = store.export_ab_report()
            print(f"A/B report written to: {path}")
        except Exception as e:
            print(f"A/B report not available: {e}")
        return

    if "--db-stats" in args:
        vids = store.fetch_all_videos()
        n = len(vids)
        chans = { (v.get("channel_id") or "").strip() for v in vids if v.get("channel_id") }
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        def _days(p):
            try:
                return (now - datetime.fromisoformat((p or "").replace("Z","+00:00"))).days
            except Exception:
                return 10**9
        last30 = sum(1 for v in vids if _days(v.get("published_at","")) <= 30)
        last90 = sum(1 for v in vids if _days(v.get("published_at","")) <= 90)
        longform = sum(1 for v in vids if int(v.get("duration_sec") or 0) >= MIN_LONG_SEC)
        print("DB STATS")
        print(f"  videos: {n}  (longform: {longform})")
        print(f"  unique channels: {len(chans)}")
        print(f"  uploaded last 30d: {last30} ({(last30/max(1,n))*100:.1f}%)")
        print(f"  uploaded last 90d: {last90} ({(last90/max(1,n))*100:.1f}%)")
        return

    # Start A/B by CLI flag
    for a in sys.argv[1:]:
        if a.startswith("--ab="):
            globals()["_ab_active"] = True
            name = a.split("=",1)[1]
            name = "".join(ch for ch in name if ch.isprintable()).strip()
            import re as _re
            name = _re.sub(r"\s+", " ", name)
            globals()["_ab_name"] = name
            try: store.log_session_event("ab_start", name=_ab_name)
            except Exception: pass
            print(f"A/B started via CLI: {_ab_name}")

    # ensure DB has something to show
    ensure_seed()

    BUFFER_K = 40 # pull a reasonably large page-set once
    buffer = None
    offset = 0

    while True:
        if buffer is None:
            buffer = recommend(k=BUFFER_K)
            offset = 0
            if not buffer:
                print("No recommendations available.")
                return

        page = buffer[offset: offset + PAGE_SIZE]
        if not page:
            buffer = None
            continue

        pick, action = pick_interactive(page)
        if pick is None and action == "next":
            offset += PAGE_SIZE
            continue
        if pick is None:
            # something changed (t/r/R/search/AB/etc.) -> rebuild buffer
            buffer = None
            continue

        # Determine A/B context for this pick
        vid_id = pick["id"]
        ab_variant = pick.get("ab_variant") or (_ab_page_map.get(vid_id) if _ab_active else None)
        exp_name = _ab_name if _ab_active else None

        if action == "skip":
            store.add_history(vid_id, dwell_sec=0, disliked=1, skipped=1, log_to_csv=False)
            if exp_name and ab_variant:
                try: store.add_ab_event(exp_name, vid_id, ab_variant, liked=0, skipped=1)
                except Exception: pass
        else:
            liked = like_feedback()
            store.add_history(vid_id, dwell_sec=0, disliked=0 if liked else 1, skipped=0, log_to_csv=True)
            if exp_name and ab_variant:
                try:
                    store.add_ab_event(exp_name, vid_id, ab_variant, liked=1 if liked else 0, skipped=0)
                    try: store.log_session_event("ab_debug", video_id=vid_id, variant=ab_variant, liked=int(bool(liked)))
                    except Exception: pass
                except Exception:
                    pass

        expand_candidates_mixed(pick["id"])

        # After an action, rebuild the buffer so watched/skipped items disappear
        buffer = None
        offset = 0


# API wrappers (used by FastAPI)
        
from typing import Optional, List, Dict

_api_bootstrapped = False

def _api_boot():
    global _api_bootstrapped
    if _api_bootstrapped:
        return
    try:
        load_env()
    except Exception:
        pass
    try:
        store.init_db()
    except Exception:
        pass
    try:
        ensure_seed()  # pull initial most-popular and write into DB
    except Exception:
        pass
    _api_bootstrapped = True

def build_page(page: int, page_size: int, variant: str) -> list[dict]:
    """
    Build a page of results using your real recommender.
    Must return dicts with keys: id, title, channel_title, duration_sec, why (list[str])
    """
    k = page * page_size * 2
    items = recommend(k=k)
    start = (page - 1) * page_size
    out = items[start:start + page_size]
    # Ensure 'why' is always present
    for it in out:
        it["why"] = it.get("why") or []
    return out


def set_trending(on):
    globals()["_trending_only"] = bool(on)

def set_category(category_id):
    globals()["_category_filter"] = category_id

if __name__ == "__main__":
    main()
