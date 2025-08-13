import sqlite3, os, time, pathlib, csv, datetime, json
import json as _json

DATA_DIR = pathlib.Path(os.getenv("YT_REC_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "app.db"
CSV_PATH = DATA_DIR / "history.csv"
SESSION_LOG = DATA_DIR / "session.log"
SUBS_CSV = DATA_DIR / "subscriptions.csv"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # --- Tables ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS videos (
      id TEXT PRIMARY KEY,
      title TEXT,
      description TEXT,
      channel_id TEXT,
      channel_title TEXT,
      tags TEXT,
      category_id TEXT,
      view_count INTEGER,
      like_count INTEGER,
      published_at TEXT,
      duration_sec INTEGER
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      video_id TEXT,
      ts INTEGER,
      dwell_sec INTEGER DEFAULT 0,
      disliked INTEGER DEFAULT 0,
      skipped INTEGER DEFAULT 0
    )""")

    # subscriptions schema
    cur.execute("""
    CREATE TABLE IF NOT EXISTS subscriptions (
      channel_id TEXT PRIMARY KEY,
      channel_title TEXT,
      channel_url TEXT,
      noted_at INTEGER
    )""")

    # blocked channels
    cur.execute("""
    CREATE TABLE IF NOT EXISTS blocked_channels (
      channel_id TEXT PRIMARY KEY,
      channel_title TEXT,
      noted_at INTEGER
    )""")

    # favorites
    cur.execute("""
    CREATE TABLE IF NOT EXISTS favorites (
      video_id TEXT PRIMARY KEY,
      channel_id TEXT,
      channel_title TEXT,
      noted_at INTEGER
    )""")

    # --- Backfills for older schemas ---
    try: cur.execute("ALTER TABLE history ADD COLUMN dwell_sec INTEGER DEFAULT 0")
    except Exception: pass
    try: cur.execute("ALTER TABLE history ADD COLUMN disliked INTEGER DEFAULT 0")
    except Exception: pass
    try: cur.execute("ALTER TABLE history ADD COLUMN skipped INTEGER DEFAULT 0")
    except Exception: pass
    try: cur.execute("ALTER TABLE videos ADD COLUMN category_id TEXT")
    except Exception: pass
    try: cur.execute("ALTER TABLE subscriptions ADD COLUMN channel_url TEXT")
    except Exception: pass
    try: cur.execute("ALTER TABLE subscriptions ADD COLUMN noted_at INTEGER")
    except Exception: pass
    # NEW: add tags column if the DB predates it
    try: cur.execute("ALTER TABLE videos ADD COLUMN tags TEXT")
    except Exception: pass

    # --- Speed indexes ---
    cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_category ON videos(category_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_history_video ON history(video_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_history_ts ON history(ts)")

    # impressions for cooldown/ignored-penalty
    cur.execute("""
    CREATE TABLE IF NOT EXISTS impressions (
      video_id TEXT PRIMARY KEY,
      n        INTEGER DEFAULT 0,
      ts       INTEGER
    )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_impressions_ts ON impressions(ts)")

    conn.commit()
    conn.close()

    # --- Ensure history CSV exists with header ---
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts","video_id","title","channel_title","dwell_sec","disliked","skipped"])


def upsert_videos(items):
    if not items:
        return
    conn = get_conn()
    cur = conn.cursor()
    for v in items:
        cur.execute("""
        INSERT INTO videos
          (id,title,description,channel_id,channel_title,tags,category_id,view_count,like_count,published_at,duration_sec)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
          title=excluded.title,
          description=excluded.description,
          channel_id=excluded.channel_id,
          channel_title=excluded.channel_title,
          tags=excluded.tags,
          category_id=excluded.category_id,
          view_count=excluded.view_count,
          like_count=excluded.like_count,
          published_at=excluded.published_at,
          duration_sec=excluded.duration_sec
        """, (
            v.get("id"),
            v.get("title") or "",
            v.get("description") or "",
            v.get("channel_id") or "",
            v.get("channel_title") or "",
            json.dumps(v.get("tags") or []),   # store tags as JSON
            v.get("category_id") or "",
            int(v.get("view_count") or 0),
            int(v.get("like_count") or 0),
            v.get("published_at") or "",
            int(v.get("duration_sec") or 0),
        ))
    conn.commit()
    conn.close()


def fetch_all_videos():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id,title,description,channel_id,channel_title,tags,category_id,
               view_count,like_count,published_at,duration_sec
        FROM videos
    """)
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "title": r[1],
            "description": r[2],
            "channel_id": r[3],
            "channel_title": r[4],
            "tags": _json.loads(r[5]) if r[5] else [],
            "category_id": r[6],
            "view_count": r[7],
            "like_count": r[8],
            "published_at": r[9],
            "duration_sec": r[10],
        })
    return out


def fetch_unwatched(limit=200):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT video_id FROM history")
    watched = {x[0] for x in cur.fetchall()}
    cur.execute("""
        SELECT id,title,description,channel_id,channel_title,tags,category_id,
               view_count,like_count,published_at,duration_sec
        FROM videos
    """)
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        if r[0] in watched:
            continue
        out.append({
            "id": r[0],
            "title": r[1],
            "description": r[2],
            "channel_id": r[3],
            "channel_title": r[4],
            "tags": json.loads(r[5]) if r[5] else [],
            "category_id": r[6],
            "view_count": r[7],
            "like_count": r[8],
            "published_at": r[9],
            "duration_sec": r[10],
        })
    return out[:limit]


def add_history(video_id, dwell_sec=0, disliked=0, skipped=0, ts_override=None, log_to_csv=True):
    ts = int(ts_override if ts_override is not None else time.time())
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO history (video_id, ts, dwell_sec, disliked, skipped) VALUES (?,?,?,?,?)",
        (video_id, ts, int(dwell_sec or 0), int(disliked or 0), int(skipped or 0))
    )
    conn.commit()
    cur.execute("SELECT title, channel_title FROM videos WHERE id=?", (video_id,))
    row = cur.fetchone()
    conn.close()

    if not log_to_csv:
        return

    title = row[0] if row else ""
    channel_title = row[1] if row else ""
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([ts, video_id, title, channel_title, int(dwell_sec or 0), int(disliked or 0), int(skipped or 0)])


def get_history(n=200):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT video_id, ts, dwell_sec, disliked, skipped FROM history ORDER BY ts DESC LIMIT ?", (n,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_watched_ids():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT video_id FROM history")
    rows = cur.fetchall()
    conn.close()
    return {r[0] for r in rows}


def get_all_watched_ids_union_csv():
    ids = set(get_all_watched_ids())
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                vid = (row.get("video_id") or "").strip()
                if vid:
                    ids.add(vid)
    except Exception:
        pass
    return ids


def history_csv_path():
    return str(CSV_PATH)


# -------- subscriptions.csv helpers (append/update/remove in-place order) --------

def _ensure_subs_csv():
    """Ensure subscriptions.csv exists with the expected header."""
    if not SUBS_CSV.exists():
        with open(SUBS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)  # quote header too
            w.writerow(["Channel Id", "Channel Url", "Channel Title"])


def _load_subs_csv_rows():
    """Load rows as a list of dicts with canonical keys (preserves CSV order)."""
    _ensure_subs_csv()
    rows = []
    with open(SUBS_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "Channel Id": (row.get("Channel Id") or "").strip(),
                "Channel Url": (row.get("Channel Url") or "").strip(),
                "Channel Title": (row.get("Channel Title") or "").strip(),
            })
    return rows

def _save_subs_csv_rows(rows):
    """Write rows back, preserving list order."""
    with open(SUBS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["Channel Id", "Channel Url", "Channel Title"],
            quoting=csv.QUOTE_ALL  # important: quote all fields so URLs aren't glued to commas
        )
        w.writeheader()
        for row in rows:
            w.writerow({
                "Channel Id": row.get("Channel Id", ""),
                "Channel Url": row.get("Channel Url", ""),
                "Channel Title": row.get("Channel Title", ""),
            })


def _append_or_update_subs_csv(channel_id, channel_title=None, channel_url=None):
    """
    Append a new line if channel_id is not present.
    If present, update that line's title/url in-place (keep original order).
    """
    if not channel_id:
        return
    rows = _load_subs_csv_rows()
    default_url = f"https://www.youtube.com/channel/{channel_id}"  # HTTPS
    updated = False
    for row in rows:
        if row["Channel Id"] == channel_id:
            if channel_title:
                row["Channel Title"] = channel_title.strip()
            if channel_url:
                row["Channel Url"] = channel_url.strip()
            if not row["Channel Url"]:
                row["Channel Url"] = default_url
            updated = True
            break
    if not updated:
        rows.append({
            "Channel Id": channel_id,
            "Channel Url": (channel_url or default_url).strip(),
            "Channel Title": (channel_title or channel_id).strip(),
        })
    _save_subs_csv_rows(rows)



def _remove_from_subs_csv(channel_id):
    """Remove the row with channel_id (if present)."""
    if not channel_id:
        return
    rows = _load_subs_csv_rows()
    new_rows = [r for r in rows if r.get("Channel Id") != channel_id]
    if len(new_rows) != len(rows):
        _save_subs_csv_rows(new_rows)


# -------- Blocked channels helpers --------

def get_blocked_channel_ids():
    """Return the set of blocked channel_ids; ensures table exists."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS blocked_channels (
      channel_id TEXT PRIMARY KEY,
      channel_title TEXT,
      noted_at INTEGER
    )""")
    conn.commit()
    cur.execute("SELECT channel_id FROM blocked_channels")
    rows = cur.fetchall()
    conn.close()
    return {r[0] for r in rows}


def block_channel(channel_id, channel_title=""):
    import time as _t
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO blocked_channels (channel_id, channel_title, noted_at) VALUES (?,?,?)",
        (channel_id, channel_title, int(_t.time()))
    )
    conn.commit()
    conn.close()

def unblock_channel(channel_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM blocked_channels WHERE channel_id=?", (channel_id,))
    conn.commit()
    conn.close()

def list_blocked_channels():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT channel_id, channel_title, noted_at FROM blocked_channels ORDER BY noted_at DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


# -------- Subscriptions --------

def upsert_subscriptions(rows):
    """rows: list[dict] with keys: channel_id, channel_title, noted_at, (optional) channel_url"""
    if not rows: return
    conn = get_conn(); cur = conn.cursor()
    for r in rows:
        cur.execute("""
            INSERT OR REPLACE INTO subscriptions (channel_id, channel_title, channel_url, noted_at)
            VALUES (?,?,?,?)
        """, (r.get("channel_id",""), r.get("channel_title",""), r.get("channel_url",""), int(r.get("noted_at") or 0)))
    conn.commit(); conn.close()

def import_subscriptions_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = []
        now = int(time.time())
        for row in r:
            rows.append({
                "channel_id": row.get("channel_id",""),
                "channel_title": row.get("channel_title",""),
                "channel_url": row.get("channel_url",""),
                "noted_at": now,
            })
    upsert_subscriptions(rows)

def get_subscribed_channel_ids():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT channel_id FROM subscriptions")
    rows = cur.fetchall(); conn.close()
    return {r[0] for r in rows}

def get_subscriptions():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS subscriptions (
      channel_id   TEXT PRIMARY KEY,
      channel_title TEXT,
      channel_url   TEXT,
      noted_at      INTEGER
    )""")
    conn.commit()
    cur.execute("SELECT channel_id, channel_title, channel_url FROM subscriptions")
    rows = cur.fetchall()
    conn.close()
    return [{"channel_id": r[0], "channel_title": r[1] or "", "channel_url": r[2] or ""} for r in rows]

# --- direct subscribe / unsubscribe helpers used by main.py ---

def add_subscription(channel_id: str, channel_title: str = None, channel_url: str = None):
    """
    Upsert a single subscription in the DB, and mirror it into data/subscriptions.csv.
    If the channel already exists in the CSV, we update its row in place; otherwise we append.
    """
    if not channel_id:
        return
    now = int(time.time())
    with get_conn() as conn:
        cur = conn.cursor()
        # ensure table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
          channel_id   TEXT PRIMARY KEY,
          channel_title TEXT,
          channel_url   TEXT,
          noted_at      INTEGER
        )""")
        # backfill title from videos if not provided
        if not channel_title:
            try:
                row = cur.execute(
                    "SELECT channel_title FROM videos WHERE channel_id=? AND COALESCE(channel_title,'')<>'' ORDER BY rowid DESC LIMIT 1",
                    (channel_id,)
                ).fetchone()
                if row and row[0]:
                    channel_title = row[0]
            except Exception:
                pass
        channel_title = (channel_title or channel_id).strip()
        cur.execute("""
            INSERT INTO subscriptions (channel_id, channel_title, channel_url, noted_at)
            VALUES (?,?,?,?)
            ON CONFLICT(channel_id) DO UPDATE SET
              channel_title = COALESCE(excluded.channel_title, subscriptions.channel_title),
              channel_url   = COALESCE(excluded.channel_url,   subscriptions.channel_url),
              noted_at      = excluded.noted_at
        """, (channel_id, channel_title, channel_url, now))
    # mirror to CSV (append if missing; update row otherwise)
    _append_or_update_subs_csv(channel_id, channel_title, channel_url)

def remove_subscription(channel_id: str):
    """Delete a subscription by channel_id (DB + subscriptions.csv)."""
    if not channel_id:
        return
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM subscriptions WHERE channel_id=?", (channel_id,))
    _remove_from_subs_csv(channel_id)


# -------- Favorites --------

def add_favorite_video(video_id, channel_id, channel_title):
    conn = get_conn(); cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO favorites(video_id, channel_id, channel_title, noted_at) VALUES (?,?,?,strftime('%s','now'))",
        (video_id, channel_id, channel_title),
    )
    conn.commit(); conn.close()

def add_favorite_channel(channel_id, channel_title):
    conn = get_conn(); cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO favorites(video_id, channel_id, channel_title, noted_at) VALUES (?,?,?,strftime('%s','now'))",
        (f"channel::{channel_id}", channel_id, channel_title),
    )
    conn.commit(); conn.close()

def list_favorites():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT video_id, channel_id, channel_title, noted_at FROM favorites ORDER BY noted_at DESC")
    rows = cur.fetchall(); conn.close()
    return rows

def remove_favorite(key):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("DELETE FROM favorites WHERE video_id = ? OR channel_id = ?", (key, key))
    conn.commit(); conn.close()


# -------- Categories roll-up (for interactive menu) --------

def get_top_categories(limit=10):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""
    SELECT COALESCE(NULLIF(category_id,''),'Unknown') AS category_id, COUNT(*) AS n
    FROM videos
    GROUP BY COALESCE(NULLIF(category_id,''),'Unknown')
    ORDER BY n DESC
    LIMIT ?
    """, (limit,))
    rows = cur.fetchall(); conn.close()
    return rows


# -------- Tags backfill helpers (new) --------

def list_ids_missing_tags(limit=500):
    conn = get_conn(); 
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT id FROM videos
        WHERE tags IS NULL OR tags='' OR tags='[]'
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [r["id"] for r in rows]

def update_tags_bulk(tag_map):
    """tag_map: dict[id] = list[str]"""
    if not tag_map:
        return
    conn = get_conn(); cur = conn.cursor()
    for vid, tags in tag_map.items():
        cur.execute("UPDATE videos SET tags=? WHERE id=?", (json.dumps(tags or []), vid))
    conn.commit(); conn.close()


# ---------- Exports (Phase 2.3/2.4) ----------

def export_blocked_csv(path=None):
    """
    Writes blocked_channels.csv with columns:
    channel_id,channel_title,noted_at,noted_at_iso
    Returns the file path.
    """
    p = path or (DATA_DIR / "blocked_channels.csv")
    rows = list_blocked_channels()  # [(channel_id, channel_title, noted_at), ...]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel_id","channel_title","noted_at","noted_at_iso"])
        for cid, ctitle, noted in rows:
            iso = datetime.datetime.utcfromtimestamp(int(noted or 0)).isoformat() + "Z" if noted else ""
            w.writerow([cid, ctitle or "", int(noted or 0), iso])
    return str(p)


def export_liked_history_csv(path=None):
    """
    Exports only 'liked' events: disliked=0 AND skipped=0 (your current semantics).
    Columns: ts,video_id,title,channel_title,dwell_sec,disliked,skipped
    """
    p = path or (DATA_DIR / "liked_history.csv")
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""
        SELECT h.ts, h.video_id, v.title, v.channel_title, h.dwell_sec, h.disliked, h.skipped
        FROM history h
        LEFT JOIN videos v ON v.id = h.video_id
        WHERE COALESCE(h.disliked,0)=0 AND COALESCE(h.skipped,0)=0
        ORDER BY h.ts ASC
    """)
    rows = cur.fetchall()
    conn.close()
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts","video_id","title","channel_title","dwell_sec","disliked","skipped"])
        for r in rows:
            w.writerow(r)
    return str(p)


def export_full_history_csv(path=None):
    """
    Exports the entire history table.
    Columns: ts,ts_iso,video_id,title,channel_title,dwell_sec,disliked,skipped
    """
    p = path or (DATA_DIR / "full_history.csv")
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""
        SELECT h.ts, h.video_id, v.title, v.channel_title, h.dwell_sec, h.disliked, h.skipped
        FROM history h
        LEFT JOIN videos v ON v.id = h.video_id
        ORDER BY h.ts ASC
    """)
    rows = cur.fetchall(); conn.close()
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts","ts_iso","video_id","title","channel_title","dwell_sec","disliked","skipped"])
        for ts, vid, title, ctitle, dwell, d, s in rows:
            iso = datetime.datetime.utcfromtimestamp(int(ts or 0)).isoformat() + "Z" if ts else ""
            w.writerow([ts, iso, vid, title or "", ctitle or "", int(dwell or 0), int(d or 0), int(s or 0)])
    return str(p)


# ---------- Optional: view_count snapshots for trending analysis ----------

def _ensure_view_counts_table(cur):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS view_counts (
      video_id   TEXT,
      ts         INTEGER,
      view_count INTEGER,
      PRIMARY KEY (video_id, ts)
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_view_counts_ts ON view_counts(ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_view_counts_vid ON view_counts(video_id)")

def snapshot_view_counts():
    """
    Take a point-in-time snapshot of current view_count values from videos table.
    Returns (inserted_rows, csv_path) for convenience.
    """
    ts_now = int(time.time())
    conn = get_conn(); cur = conn.cursor()
    _ensure_view_counts_table(cur)
    cur.execute("SELECT id, view_count FROM videos")
    rows = cur.fetchall()

    # Insert snapshot
    ins = 0
    for vid, vc in rows:
        try:
            cur.execute(
                "INSERT OR IGNORE INTO view_counts(video_id, ts, view_count) VALUES (?,?,?)",
                (vid, ts_now, int(vc or 0))
            )
            ins += cur.rowcount
        except Exception:
            pass
    conn.commit()

    # Also write a CSV for one-off analysis
    csv_path = DATA_DIR / f"view_counts_{ts_now}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id","ts","view_count"])
        cur.execute("SELECT video_id, ts, view_count FROM view_counts WHERE ts=?", (ts_now,))
        for r in cur.fetchall():
            w.writerow(r)

    conn.close()
    return ins, str(csv_path)


# ---------- Telemetry (session log) ----------

def log_session_page(k, trend_only, category_id, temp_blocked_cnt, candidate_count):
    """
    Append a simple line to data/session.log so you can analyze later.
    """
    ts = int(time.time())
    iso = datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z"
    with open(SESSION_LOG, "a", encoding="utf-8") as f:
        f.write(
            f"{iso}\tk={k}\ttrend={'1' if trend_only else '0'}\tcat={category_id or ''}\t"
            f"temp_blocked={int(temp_blocked_cnt or 0)}\tcands={int(candidate_count or 0)}\n"
        )


# ---------- Telemetry ----------
def log_session_event(event, **fields):
    """
    Append one JSON line per event to data/session.log.
    event: short string like 'page', 'why', 'trace', 'ab_start', 'ab_stop'
    fields: any JSON-serializable data
    """
    p = DATA_DIR / "session.log"
    row = {"ts": int(time.time()), "event": event, **fields}
    with open(p, "a", encoding="utf-8") as f:
        f.write(_json.dumps(row, ensure_ascii=False) + "\n")
    return str(p)

# ---------- A/B events ----------
def _ensure_ab_tables(cur):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ab_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      experiment TEXT,
      ts INTEGER,
      video_id TEXT,
      variant TEXT,   -- "A" or "B"
      liked INTEGER,
      skipped INTEGER
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ab_events_exp ON ab_events(experiment)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ab_events_ts ON ab_events(ts)")

def add_ab_event(experiment, video_id, variant, liked, skipped, ts=None):
    """Legacy helper; kept for compatibility."""
    conn = get_conn(); cur = conn.cursor()
    _ensure_ab_tables(cur)
    cur.execute("""
        INSERT INTO ab_events (experiment, ts, video_id, variant, liked, skipped)
        VALUES (?,?,?,?,?,?)
    """, (experiment, int(ts or time.time()), video_id, variant, int(liked or 0), int(skipped or 0)))
    conn.commit(); conn.close()

def export_ab_report(experiment=None, path=None):
    """
    Aggregate per-experiment winrates (variant A vs B).
    Returns CSV path.
    """
    p = path or (DATA_DIR / "ab_report.csv")
    conn = get_conn(); cur = conn.cursor()
    _ensure_ab_tables(cur)
    q = "SELECT experiment, variant, SUM(COALESCE(liked,0)) AS likes, SUM(COALESCE(skipped,0)) AS skips, COUNT(*) AS n FROM ab_events"
    if experiment:
        q += " WHERE experiment=?"
        cur.execute(q + " GROUP BY experiment, variant", (experiment,))
    else:
        cur.execute(q + " GROUP BY experiment, variant")
    rows = cur.fetchall(); conn.close()

    # collapse to A vs B per experiment
    from collections import defaultdict
    agg = defaultdict(lambda: {"A": {"likes":0,"skips":0,"n":0}, "B": {"likes":0,"skips":0,"n":0}})
    for exp, var, likes, skips, n in rows:
        agg[exp][var]["likes"] = likes or 0
        agg[exp][var]["skips"] = skips or 0
        agg[exp][var]["n"] = n or 0

    import csv
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment","variant","likes","skips","n","like_rate"])
        for exp, d in agg.items():
            for var in ("A","B"):
                x = d[var]
                like_rate = (x["likes"] / max(1, x["n"])) if x["n"] else 0.0
                w.writerow([exp, var, x["likes"], x["skips"], x["n"], f"{like_rate:.4f}"])
    return str(p)

# --- Simple A/B logging helpers (choice + feedback) ---
def ab_log_choice(experiment, variant, video_id, ts=None):
    """Record that a variant won the impression (before user feedback)."""
    if not (experiment and variant and video_id): return
    conn = get_conn(); cur = conn.cursor()
    _ensure_ab_tables(cur)
    cur.execute("""
        INSERT INTO ab_events (experiment, ts, video_id, variant, liked, skipped)
        VALUES (?,?,?,?,0,0)
    """, (experiment, int(ts or time.time()), video_id, variant))
    conn.commit(); conn.close()

def ab_log_feedback(experiment, variant, video_id, liked, skipped=0):
    """Update the latest choice row for (exp,variant,video_id) with the outcome."""
    if not (experiment and variant and video_id): return
    conn = get_conn(); cur = conn.cursor()
    _ensure_ab_tables(cur)
    cur.execute("""
        UPDATE ab_events SET liked=?, skipped=?
        WHERE id = (
          SELECT id FROM ab_events
          WHERE experiment=? AND variant=? AND video_id=?
          ORDER BY id DESC LIMIT 1
        )
    """, (int(liked or 0), int(skipped or 0), experiment, variant, video_id))
    conn.commit(); conn.close()

# ---------- Impressions (cooldown + ignored-penalty) ----------

def log_impressions(video_ids):
    """Increment impression counters and last-seen timestamp for each id."""
    if not video_ids: return
    now = int(time.time())
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS impressions (
            video_id TEXT PRIMARY KEY, n INTEGER DEFAULT 0, ts INTEGER)""")
        for vid in set(video_ids):
            cur.execute("""
                INSERT INTO impressions(video_id, n, ts)
                VALUES(?, 1, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                  n = COALESCE(n,0) + 1,
                  ts = excluded.ts
            """, (vid, now))

def get_recently_impressed_ids(ttl_hours, min_n=1):
    """Return ids impressed within the last ttl_hours (and at least min_n total)."""
    horizon = int(time.time()) - int(ttl_hours or 0) * 3600
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT video_id FROM impressions WHERE ts >= ? AND COALESCE(n,0) >= ?",
            (horizon, int(min_n or 1))
        ).fetchall()
    return {r[0] for r in rows}

def get_impression_like_counts(video_ids):
    """Return {vid: (impressions, likes)} for the given IDs."""
    if not video_ids: return {}
    ids = list({v for v in video_ids if v})
    q = ",".join("?" for _ in ids)
    with get_conn() as conn:
        imp = conn.execute(f"""
            SELECT video_id, COALESCE(n,0) FROM impressions
            WHERE video_id IN ({q})
        """, ids).fetchall()
        imp_map = {r[0]: int(r[1] or 0) for r in imp}
        # treat "liked" as (disliked=0 AND skipped=0)
        lk = conn.execute(f"""
            SELECT video_id, SUM(CASE WHEN COALESCE(disliked,0)=0 AND COALESCE(skipped,0)=0 THEN 1 ELSE 0 END)
            FROM history WHERE video_id IN ({q}) GROUP BY video_id
        """, ids).fetchall()
        like_map = {r[0]: int(r[1] or 0) for r in lk}
    return {vid: (imp_map.get(vid, 0), like_map.get(vid, 0)) for vid in ids}

# ---------- Export full history (telemetry) ----------
def export_history_csv(path=None):
    """
    Full history export (ASC time) with both epoch + ISO.
    """
    p = path or (DATA_DIR / "history_full.csv")
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""
        SELECT h.ts, h.video_id, v.title, v.channel_title, h.dwell_sec, h.disliked, h.skipped
        FROM history h LEFT JOIN videos v ON v.id = h.video_id
        ORDER BY h.ts ASC
    """)
    rows = cur.fetchall(); conn.close()
    import csv
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts","ts_iso","video_id","title","channel_title","dwell_sec","disliked","skipped"])
        for ts, vid, title, ctitle, dwell, dis, sk in rows:
            iso = datetime.datetime.utcfromtimestamp(int(ts or 0)).isoformat() + "Z"
            w.writerow([ts, iso, vid, title or "", ctitle or "", int(dwell or 0), int(dis or 0), int(sk or 0)])
    return str(p)
