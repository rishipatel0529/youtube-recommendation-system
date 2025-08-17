"""
embed_text.py — Compute text embeddings for video metadata and persist to SQLite.

- Uses SentenceTransformers (default: all-MiniLM-L6-v2) to embed titles + descriptions
- Stores embeddings as float32 BLOBs in video_text_embeds table
- Safe to re-run: uses UPSERT to update existing rows
- Supports batching, model selection, overwrite mode, and quick test limits
- Entry point: `python -m src.dl.embed_text --model ... --batch ...`

This script underpins semantic search & recommendation ranking by enabling
vector-based retrieval directly from the local SQLite store.
"""

from __future__ import annotations
import argparse, datetime as dt, sqlite3, sys, os
from typing import Iterable, Tuple, Optional, List
import numpy as np
from tqdm import tqdm

try:
    from src import store
    def get_conn() -> sqlite3.Connection:
        return store.get_conn()
except Exception:
    def get_conn() -> sqlite3.Connection:
        db_path = os.getenv("YTCLI_DB", "data/store.sqlite3")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

def _videos_iter(cur: sqlite3.Cursor, limit: Optional[int]=None) -> Iterable[Tuple[str,str,str]]:
    # Yield (video_id, title, desc) from videos table with fallback column names.
    cols = [r[1] for r in cur.execute("PRAGMA table_info(videos)").fetchall()]
    id_col = "id" if "id" in cols else ("video_id" if "video_id" in cols else None)
    title_col = "title" if "title" in cols else None
    desc_col = "description" if "description" in cols else ("desc" if "desc" in cols else None)
    if not id_col or not title_col:
        raise RuntimeError(f"videos table needs id/video_id and title. found: {cols}")

    q = f"SELECT {id_col}, {title_col}" + (f", {desc_col}" if desc_col else ", ''") + " FROM videos"
    if limit:
        q += f" LIMIT {int(limit)}"
    for row in cur.execute(q):
        vid, title, desc = (row[0], row[1], row[2] if len(row) > 2 else "")
        yield vid, title or "", desc or ""

def _ensure_table(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS video_text_embeds (
      video_id TEXT PRIMARY KEY,
      dim INTEGER NOT NULL,
      model TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      emb BLOB NOT NULL
    )
    """)
    conn.commit()

def _existing_ids(cur: sqlite3.Cursor, model_name: str) -> set:
    cur.execute("SELECT video_id FROM video_text_embeds WHERE model=?", (model_name,))
    return {r[0] for r in cur.fetchall()}

def _encode_texts(texts: List[str], model_name: str, batch: int) -> np.ndarray:
    # Batch-encode texts into normalized embeddings using SentenceTransformers.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    embs = []
    for i in tqdm(range(0, len(texts), batch), desc="Encoding"):
        embs.append(model.encode(texts[i:i+batch], normalize_embeddings=True))
    return np.vstack(embs)

def main():
    # CLI entrypoint: parse args, fetch videos, encode texts, and write embeddings.
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="For quick tests")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if already present for this model")
    args = ap.parse_args()

    conn = get_conn(); cur = conn.cursor()
    _ensure_table(conn)

    vids, texts = [], []
    existing = set() if args.overwrite else _existing_ids(cur, args.model)
    for vid, title, desc in _videos_iter(cur, limit=args.limit):
        if not args.overwrite and vid in existing: 
            continue
        text = (title + " [SEP] " + (desc[:1000] if desc else "")).strip()
        if not text:
            text = title or "untitled"
        vids.append(vid)
        texts.append(text)

    if not vids:
        print("[embed_text] nothing to do.")
        return

    embs = _encode_texts(texts, args.model, args.batch)
    now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    dim = int(embs.shape[1])

    with conn:
        for vid, vec in tqdm(list(zip(vids, embs)), desc="Writing to DB"):
            conn.execute("""
            INSERT INTO video_text_embeds(video_id, dim, model, updated_at, emb)
            VALUES(?,?,?,?,?)
            ON CONFLICT(video_id) DO UPDATE SET dim=excluded.dim, model=excluded.model,
                  updated_at=excluded.updated_at, emb=excluded.emb
            """, (vid, dim, args.model, now, memoryview(np.asarray(vec, dtype=np.float32).tobytes())))
    print(f"[embed_text] Stored {len(vids)} embeddings (dim={dim}) for model={args.model}")

if __name__ == "__main__":
    main()