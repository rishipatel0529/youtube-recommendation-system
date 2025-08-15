import streamlit as st, requests, os
from html import escape

# --- API base ---
try:
    API = st.secrets["API_URL"]
except Exception:
    API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="YT Recs", layout="wide")


# --- YouTube-like typography (font + sizes) ---
st.markdown("""
<style>
  /* Load Roboto like YouTube */
  @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

  :root{
    --yt-title-size: 1.0rem;   /* YouTube-ish title size */
    --yt-meta-size: 0.9rem;    /* meta/caption size */
  }

/* App-wide text — avoid clobbering icon fonts */
html, body, [data-testid="stAppViewContainer"], .stMarkdown, .stText,
.stSubheader, .stCaption {
  font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Arial, sans-serif !important;
}

/* Bring back Material Symbols for any icon elements */
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');
.material-icons, span[class*="material-icons"], i[class*="material-icons"] {
  font-family: 'Material Symbols Outlined' !important;
  font-weight: normal !important;
  font-style: normal !important;
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}


  /* Video titles (we'll render them as h3) */
  h3{
    font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Arial, sans-serif !important;
    font-size: var(--yt-title-size) !important;
    font-weight: 500 !important;          /* YouTube uses medium */
    line-height: 1.4 !important;
    margin: .35rem 0 .25rem 0 !important;
  }

/* Channel / meta line (Streamlit caption) — 1-line clamp */
.stCaption, .stCaption p{
  font-size: var(--yt-meta-size) !important;
  color: #aaa !important;
  margin-top: .15rem !important;

  /* clamp to one line */
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  display: block; /* ensures ellipsis works */
}


  /* Buttons — match current Streamlit structure */
  div[data-testid="stButton"] > button,
  button[kind],
  button[data-baseweb="button"],
  [data-testid="baseButton-primary"],
  [data-testid="baseButton-secondary"]{
    font-size: 12px !important;      /* adjust to your taste */
    padding: .35rem .75rem !important;
    border-radius: 12px !important;
    line-height: 1.1 !important;
  }

  /* Submit buttons inside forms */
  div[data-testid="stFormSubmitButton"] > button{
    font-size: 12px !important;
    padding: .35rem .75rem !important;
    border-radius: 12px !important;
  }
            
/* Clamp video titles to 2 lines like YouTube */
.yt-title{
  font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Arial, sans-serif !important;
  font-size: var(--yt-title-size) !important;
  font-weight: 500 !important;
  line-height: 1.35 !important;

  /* the magic */
  display: -webkit-box;
  -webkit-line-clamp: 2;       /* show up to 2 lines */
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;

  /* reserve space so every card is same height */
  min-height: calc(2 * 1.35 * var(--yt-title-size));
  margin: .35rem 0 .25rem 0 !important;
}
            
/* Meta row: truncate channel only, keep rest visible on one line */
.yt-meta{
  display:flex; align-items:center;
  gap:.35rem;
  color:#aaa;
  font-size: var(--yt-meta-size);
  margin-top:.15rem;
}
.yt-meta .ch{
  flex:1; min-width:0;      /* allow ellipsis to work */
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.yt-meta .rest{
  white-space:nowrap;       /* never wraps */
}


</style>
""", unsafe_allow_html=True)



# --- session state ---
if "variant" not in st.session_state: st.session_state.variant = "A"
if "page" not in st.session_state: st.session_state.page = 1
if "trending" not in st.session_state: st.session_state.trending = False
if "subs_set" not in st.session_state: st.session_state.subs_set = set()
if "category_id" not in st.session_state: st.session_state.category_id = None
# caching + per-card state
if "items_cache" not in st.session_state: st.session_state.items_cache = None
if "items_cache_key" not in st.session_state: st.session_state.items_cache_key = None
if "needs_reload" not in st.session_state: st.session_state.needs_reload = True
if "pending_feedback" not in st.session_state: st.session_state.pending_feedback = {}  # {video_id: True}

# --- helpers ---
def api_get(path, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=30); r.raise_for_status()
    return r.json()

def api_post(path, payload):
    r = requests.post(f"{API}{path}", json=payload, timeout=60); r.raise_for_status()
    return r.json()

def load_subs_set():
    try:
        resp = api_get("/subs")
        st.session_state.subs_set = {x["channel_id"] for x in resp.get("items", [])}
    except Exception:
        st.session_state.subs_set = set()

def load_categories():
    try:
        resp = api_get("/categories")
        return resp.get("items", [])
    except Exception:
        return []

def fmt_duration(sec):
    """Return m:ss (or h:mm:ss if >= 1 hour) from seconds."""
    try:
        total = int(sec or 0)
    except Exception:
        total = 0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def trunc_channel(name: str, hard_max: int = 39, keep: int = 36) -> str:
    """Hard truncate channel names: if len>hard_max, keep first `keep` and add '...'."""
    s = (name or "").strip()
    return (s[:keep].rstrip() + "...") if len(s) > hard_max else s


def fetch_items(mode, q, page, page_size):
    q = (q or "").strip()
    if q:  # if user typed anything, run search
        return api_post("/search", {"q": q, "page": page, "page_size": page_size})
    else:
        return api_post("/recommend", {"user_id":"local","page": page,"page_size": page_size,"variant": st.session_state.variant})

# --- pre-widget triggers (must run before widgets are created) ---
if st.session_state.get("_clear_q_trigger"):
    st.session_state.pop("q", None)
    st.session_state["_clear_q_trigger"] = False

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Controls")

    # Variant selector
    sel_variant = st.selectbox("Variant", ["A", "B"], index=0 if st.session_state.variant == "A" else 1)
    if sel_variant != st.session_state.variant:
        st.session_state.variant = sel_variant
        st.session_state.needs_reload = True

    mode = st.radio("Mode", ["Recommend", "Search"], index=0, horizontal=True)

    # Query + compact clear (aligned ✕)
    st.markdown("**Query**")
    qcol, clrcol = st.columns([8, 1])
    with qcol:
        st.text_input(
            "Query",
            key="q",
            value=st.session_state.get("q", ""),
            placeholder="Search videos…",
            label_visibility="collapsed",
        )
    with clrcol:
        # no spacer -> aligns with the input vertically
        if st.button("✕", key="clear_q", help="Clear query", use_container_width=True):
            st.session_state["_clear_q_trigger"] = True
            st.session_state.needs_reload = True
            st.rerun()

    page_size = st.slider("Page size", 5, 20, 10)

    # Refresh button
    if st.button("Refresh"):
        try:
            api_post("/control/refresh", {})
        except Exception:
            pass
        st.session_state.needs_reload = True
        st.rerun()

    # Trending only — circle style via radio
    trend_choice = st.radio(
        "Trending only",
        ["On", "Off"],
        index=0 if st.session_state.trending else 1,
        horizontal=True,
    )
    new_trend = (trend_choice == "On")
    if new_trend != st.session_state.trending:
        st.session_state.trending = new_trend
        try:
            api_post("/control/trending", {"on": bool(new_trend)})
            api_post("/control/refresh", {})
        except Exception:
            pass
        st.session_state.needs_reload = True
        st.rerun()

    # Category section
    st.markdown("### Category")
    cats = load_categories()
    names = ["All"] + [f"{c['name']} ({c['id'] or '—'})" for c in cats]
    pick = st.selectbox("Pick category", names, index=0)
    new_cat_id = None if pick == "All" else cats[names.index(pick) - 1]["id"]
    if new_cat_id != st.session_state.category_id:
        st.session_state.category_id = new_cat_id
        try:
            api_post("/control/category", {"category_id": new_cat_id})
            api_post("/control/refresh", {})
        except Exception:
            pass
        st.session_state.needs_reload = True

    # Pager
    c1, c2 = st.columns(2)
    if c1.button("Prev"):
        st.session_state.page = max(1, st.session_state.page - 1)
        st.session_state.needs_reload = True
    if c2.button("Next"):
        st.session_state.page += 1
        st.session_state.needs_reload = True
    st.caption(f"Page {st.session_state.page}")

# subs list (id set used by buttons)
load_subs_set()

# --- read query from session ---
q = st.session_state.get("q", "")

# --- page cache key + conditional fetch ---
cache_key = (
    st.session_state.variant,
    mode,
    (q or "").strip(),
    st.session_state.page,
    page_size,
    bool(st.session_state.trending),
    st.session_state.category_id,
)

if st.session_state.needs_reload or cache_key != st.session_state.items_cache_key:
    items = fetch_items(mode, q, st.session_state.page, page_size)
    st.session_state.items_cache = items
    st.session_state.items_cache_key = cache_key
    st.session_state.needs_reload = False
else:
    items = st.session_state.items_cache or []

# --- render items (stable after Watch) ---
cols = st.columns(3)
for i, it in enumerate(items):
    c = cols[i % 3]
    with c:
        title = it.get("title", "")
        channel_name = it.get("channel", "")
        vid = it.get("video_id")
        ch_id = it.get("channel_id") or ""
        duration = it.get("duration_sec", 0)
        rank = it.get("rank")
        variant = it.get("variant", st.session_state.variant)

        # Clickable thumbnail that opens YouTube in a new tab
        if vid:
            thumb_base = f"https://i.ytimg.com/vi/{vid}"
            thumb_url  = f"{thumb_base}/mqdefault.jpg"
            thumb_html = f"""
            <a href="https://www.youtube.com/watch?v={vid}" target="_blank" rel="noopener">
            <img
                src="{thumb_url}"
                srcset="{thumb_base}/hq720.jpg 1280w, {thumb_url} 320w"
                sizes="(max-width: 800px) 100vw, 33vw"
                style="width:100%; aspect-ratio:16/9; object-fit:cover; display:block; border-radius:12px;"
                alt="{escape(title)}"
            />
            </a>
            """
            st.markdown(thumb_html, unsafe_allow_html=True)

        st.markdown(f'<div class="yt-title">{escape(title)}</div>', unsafe_allow_html=True)

        ch_display = trunc_channel(channel_name)  # hard-limit the channel name

        meta_html = f'''
        <div class="yt-meta">
        <span class="ch">{escape(ch_display)}</span>
        <span class="rest">• {fmt_duration(duration)} • #{rank} • {variant}</span>
        </div>
        '''
        st.markdown(meta_html, unsafe_allow_html=True)

        # Subscribe / Unsubscribe
        sc1, _ = st.columns(2)
        if ch_id:
            if ch_id in st.session_state.subs_set:
                if sc1.button("Unsubscribe", key=f"unsub_{ch_id}_{i}"):
                    api_post("/subs/unsubscribe", {"channel_id": ch_id})
                    load_subs_set()
                    st.toast("Unsubscribed")
            else:
                if sc1.button("Subscribe", key=f"sub_{ch_id}_{i}"):
                    api_post("/subs/subscribe", {"channel_id": ch_id, "channel_title": channel_name})
                    load_subs_set()
                    st.toast("Subscribed")
        else:
            sc1.button("Subscribe", key=f"sub_disabled_{i}", disabled=True)

        # Watch -> Like/Dislike
        watched = st.session_state.pending_feedback.get(vid, False)
        if not watched:
            b1, b2 = st.columns(2)
            if b1.button("Watch", key=f"w_{vid}"):
                api_post("/interact", {
                    "user_id": "local", "video_id": vid, "event": "watch",
                    "rank": rank, "variant": variant,
                    "page": st.session_state.page,
                    "is_search": 1 if mode == "Search" else 0,
                    "query": q if mode == "Search" else None
                })
                st.session_state.pending_feedback[vid] = True
                st.rerun()
            if b2.button("Skip", key=f"s_{vid}"):
                api_post("/interact", {
                    "user_id": "local", "video_id": vid, "event": "skip",
                    "rank": rank, "variant": variant,
                    "page": st.session_state.page,
                    "is_search": 1 if mode == "Search" else 0,
                    "query": q if mode == "Search" else None
                })
                st.session_state.needs_reload = True
                st.rerun()
        else:
            st.caption("Watched. How was it?")
            lb, db = st.columns(2)
            if lb.button("👍 Like", key=f"like_{vid}"):
                api_post("/interact", {
                    "user_id": "local", "video_id": vid, "event": "like",
                    "rank": rank, "variant": variant,
                    "page": st.session_state.page,
                    "is_search": 1 if mode == "Search" else 0,
                    "query": q if mode == "Search" else None
                })
                st.session_state.pending_feedback.pop(vid, None)
                st.session_state.needs_reload = True
                st.rerun()
            if db.button("👎 Dislike", key=f"dislike_{vid}"):
                api_post("/interact", {
                    "user_id": "local", "video_id": vid, "event": "dislike",
                    "rank": rank, "variant": variant,
                    "page": st.session_state.page,
                    "is_search": 1 if mode == "Search" else 0,
                    "query": q if mode == "Search" else None
                })
                st.session_state.pending_feedback.pop(vid, None)
                st.session_state.needs_reload = True
                st.rerun()

        # '...' menu (reordered: Save video → Save channel → Block channel)
        with st.expander("⋯", expanded=False):
            m1, m2, m3 = st.columns(3)
            if m1.button("Save video", key=f"favv_{vid}"):
                api_post("/action/save_video", {"video_id": vid})
                st.toast("Saved video")
            if m2.button("Save channel", key=f"favc_{ch_id}_{i}", disabled=not bool(ch_id)):
                api_post("/action/save_channel", {"channel_id": ch_id, "channel_title": channel_name})
                st.toast("Saved channel")
            if m3.button("Block channel", key=f"blk_{ch_id}_{i}", disabled=not bool(ch_id)):
                api_post("/action/block", {"channel_id": ch_id, "channel_title": channel_name})
                st.success("Channel blocked. Refreshing…")
                try:
                    api_post("/control/refresh", {})
                except Exception:
                    pass
                st.session_state.needs_reload = True
                st.rerun()

        if it.get("why"):
            st.caption("Why: " + ", ".join(it["why"]))

# --- AB report ---
st.divider()
st.subheader("A/B Report")
since = st.text_input("Since (YYYY-MM-DD)", value="2025-01-01")
if st.button("Refresh Report"):
    r = api_get("/ab/report", since=since)
    st.json(r)
