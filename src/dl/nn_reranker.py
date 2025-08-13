"""
Neural re-ranker that scores (user, item) pairs using:
- user vector = mean of embeddings of liked/favorited videos
- item vector = text embedding from video_text_embeds
- features = [user_dot_item, item_vec] (+ optional view-count)

Training is pointwise (BCE). You can train/eval and then use it to re-rank any candidate list.

Usage:
  # Train (uses your DB + log tables; falls back to favorites if likes are scarce)
  python -m src.dl.nn_reranker --epochs 5 --lr 1e-3

  # Score a list of video_ids (comma-separated) and print sorted by score desc
  python -m src.dl.nn_reranker --predict v123,v456,v789

Artifacts:
  data/nn_reranker.pt   (model)
  data/nn_config.json   (dim, model name, feature recipe)
"""
from __future__ import annotations
import argparse, os, json, sqlite3, random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ===== DB glue =====

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

def _fetch_embed_map(conn: sqlite3.Connection, model_name: Optional[str]=None) -> Tuple[Dict[str,np.ndarray], int, str]:
    cur = conn.cursor()
    if model_name:
        cur.execute("SELECT video_id, dim, model, emb FROM video_text_embeds WHERE model=?", (model_name,))
    else:
        # pick the most recent model in the table
        cur.execute("""
            SELECT video_id, dim, model, emb FROM video_text_embeds
            WHERE updated_at = (SELECT MAX(updated_at) FROM video_text_embeds)
        """)
    rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No embeddings found. Run: python -m src.dl.embed_text")
    embs, dim, model = {}, None, None
    for vid, d, m, blob in rows:
        if dim is None: dim = int(d); model = m
        vec = np.frombuffer(blob, dtype=np.float32)
        if vec.size != dim:
            continue
        embs[vid] = vec
    assert dim is not None and model is not None
    return embs, dim, model

def _column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def _fetch_views(conn: sqlite3.Connection) -> Dict[str, float]:
    cur = conn.cursor()
    cols = [r[1] for r in cur.execute("PRAGMA table_info(videos)").fetchall()]
    id_col = "id" if "id" in cols else ("video_id" if "video_id" in cols else None)
    if not id_col: return {}
    views_col = "views" if "views" in cols else None
    if not views_col:
        return {}
    out = {}
    for vid, v in cur.execute(f"SELECT {id_col}, {views_col} FROM videos"):
        try:
            out[vid] = float(v or 0.0)
        except Exception:
            out[vid] = 0.0
    # log scale to stabilize
    for k in out:
        out[k] = np.log1p(out[k])
    return out

# ===== User feedback mining =====

def _fetch_positive_negative_ids(conn: sqlite3.Connection) -> Tuple[List[str], List[str]]:
    """
    Tries several common places to find labels:
    - interactions/feedback table with label in {'Y','N'} or {1,0}
    - ab_events table used by your A/B harness
    - favorites/blocked as a weak proxy (favorites => positive)
    """
    cur = conn.cursor()

    # Try interactions
    for table in ["interactions", "feedback", "user_feedback", "events"]:
        if not _table_exists(conn, table): 
            continue
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
        vid_col = "video_id" if "video_id" in cols else ("id" if "id" in cols else None)
        lab_col = None
        for c in ["label","y","liked","vote","feedback","action"]:
            if c in cols: lab_col = c; break
        if vid_col and lab_col:
            pos, neg = [], []
            for vid, lab in cur.execute(f"SELECT {vid_col}, {lab_col} FROM {table}"):
                if lab in (1, "1", "Y", "y", "like", "liked", "YES", "yes"): pos.append(str(vid))
                elif lab in (0, "0", "N", "n", "dislike", "NO", "no"):      neg.append(str(vid))
            if pos or neg:
                return pos, neg

    # Try AB log (schema: ab_events with fields video_id, label maybe)
    if _table_exists(conn, "ab_events"):
        cols = [r[1] for r in cur.execute("PRAGMA table_info(ab_events)").fetchall()]
        if "video_id" in cols and "label" in cols:
            pos, neg = [], []
            for vid, lab in cur.execute("SELECT video_id, label FROM ab_events"):
                if lab in ("Y", "y", 1, "1"): pos.append(str(vid))
                elif lab in ("N", "n", 0, "0"): neg.append(str(vid))
            if pos or neg:
                return pos, neg

    # Fallback: favorites as positives; blocked as negatives
    pos, neg = [], []
    if _table_exists(conn, "favorites"):
        for (vid,) in cur.execute("SELECT video_id FROM favorites"):
            pos.append(str(vid))
    if _table_exists(conn, "blocked"):
        for (vid,) in cur.execute("SELECT video_id FROM blocked"):
            neg.append(str(vid))
    return pos, neg

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

# ===== Model =====

class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

@dataclass
class NNConfig:
    dim: int
    use_views: bool
    model_name: str

# ===== Utilities =====

def _build_user_vector(pos_ids: List[str], embed_map: Dict[str,np.ndarray], dim: int) -> np.ndarray:
    vecs = [embed_map[vid] for vid in pos_ids if vid in embed_map]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    v = np.mean(np.vstack(vecs), axis=0)
    # normalize
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)

# ===== Dataset =====

def _make_examples(pos_ids: List[str], neg_ids: List[str], embed_map: Dict[str,np.ndarray],
                   user_vec: np.ndarray, views: Dict[str, float], use_views: bool) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for vid in pos_ids:
        if vid not in embed_map: continue
        iv = embed_map[vid]
        dot = float(np.dot(user_vec, iv))
        if use_views:
            X.append(np.concatenate(([dot], iv, [views.get(vid, 0.0)],), axis=0))
        else:
            X.append(np.concatenate(([dot], iv), axis=0))
        y.append(1.0)
    for vid in neg_ids:
        if vid not in embed_map: continue
        iv = embed_map[vid]
        dot = float(np.dot(user_vec, iv))
        if use_views:
            X.append(np.concatenate(([dot], iv, [views.get(vid, 0.0)],), axis=0))
        else:
            X.append(np.concatenate(([dot], iv), axis=0))
        y.append(0.0)
    if not X:
        return np.zeros((0, 1+next(iter(embed_map.values())).shape[0]), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y

# ===== Training / Eval / Predict =====

def train(epochs: int=5, lr: float=1e-3, model_name: Optional[str]=None, seed: int=42, use_views: bool=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    conn = get_conn()
    embed_map, dim, resolved_model = _fetch_embed_map(conn, model_name)
    views = _fetch_views(conn) if use_views else {}
    pos_ids, neg_ids = _fetch_positive_negative_ids(conn)

    # If we don't have enough explicit negatives, sample from the unlabeled pool
    if len(neg_ids) < 5:
        all_ids = list(embed_map.keys())
        pool = [vid for vid in all_ids if vid not in set(pos_ids)]
        k = max(5, min(len(pool), 5 * max(1, len(pos_ids))))
        if k > 0:
            neg_sample = random.sample(pool, k)
            neg_ids = neg_ids + neg_sample
            print(f"[bootstrap] sampled {len(neg_sample)} implicit negatives from unlabeled pool.")

    print(f"[data] pos={len(pos_ids)}  neg={len(neg_ids)}  embed_dim={dim}  model={resolved_model}")
    if len(pos_ids) < 5 or len(neg_ids) < 5:
        print("[warn] Few labels found; training may be weak. Collect more likes/dislikes.")

    # Build user vector from positives (favorites/likes). If none, zero.
    uvec = _build_user_vector(pos_ids, embed_map, dim)

    # Make dataset
    X, y = _make_examples(pos_ids, neg_ids, embed_map, uvec, views, use_views)
    if not (np.any(y == 1.0) and np.any(y == 0.0)):
        raise RuntimeError("Need both positive and negative labels. Like a few and dislike a few, or enable implicit negatives.")

    # ===== Stratified split with fallback ensuring both classes in val =====
    if np.unique(y).size > 1:
        try:
            Xtr, Xva, ytr, yva = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )
        except ValueError:
            # Manual fallback: guarantee at least 1 pos/neg in val
            pos_idx = np.where(y == 1.0)[0]
            neg_idx = np.where(y == 0.0)[0]
            rng = np.random.default_rng(seed)
            n_pos_val = max(1, int(0.2 * len(pos_idx)))
            n_neg_val = max(1, int(0.2 * len(neg_idx)))
            val_pos = rng.choice(pos_idx, size=min(n_pos_val, len(pos_idx)), replace=False)
            val_neg = rng.choice(neg_idx, size=min(n_neg_val, len(neg_idx)), replace=False)
            va = np.concatenate([val_pos, val_neg])
            tr = np.setdiff1d(np.arange(len(y)), va, assume_unique=False)
            Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    else:
        # Shouldn't happen due to check above; simple split as last resort
        idx = np.arange(len(y)); np.random.shuffle(idx)
        split = int(0.8 * len(idx))
        tr, va = idx[:split], idx[split:]
        Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

    model = MLP(X.shape[1])
    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    Xv = torch.from_numpy(Xva); yv = torch.from_numpy(yva)

    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(Xt.shape[0])
        total = 0.0
        for i in range(0, Xt.shape[0], 256):
            idxb = perm[i:i+256]
            xb, yb = Xt[idxb], yt[idxb]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idxb)
        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_auc = roc_auc_score(yv.numpy(), torch.sigmoid(val_logits).numpy()) if len(yv.unique())>1 else float("nan")
        print(f"[ep {ep}] train_loss={total/len(Xt):.4f}  val_auc={val_auc:.4f}")

    os.makedirs("data", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": X.shape[1]}, "data/nn_reranker.pt")
    with open("data/nn_config.json", "w") as f:
        json.dump({"dim": dim, "use_views": use_views, "model_name": resolved_model}, f)
    print("[save] data/nn_reranker.pt  and  data/nn_config.json")

def _load_model() -> Tuple[MLP, Dict]:
    ckpt = torch.load("data/nn_reranker.pt", map_location="cpu")
    model = MLP(ckpt["input_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with open("data/nn_config.json") as f:
        cfg = json.load(f)
    return model, cfg

def predict(video_ids: List[str]) -> List[Tuple[str, float]]:
    model, cfg = _load_model()
    conn = get_conn()
    embed_map, dim, _ = _fetch_embed_map(conn, cfg["model_name"])
    views = _fetch_views(conn) if cfg.get("use_views", True) else {}
    pos_ids, _ = _fetch_positive_negative_ids(conn)
    uvec = _build_user_vector(pos_ids, embed_map, dim)

    rows = []
    for vid in video_ids:
        if vid not in embed_map: 
            continue
        iv = embed_map[vid]
        dot = float(np.dot(uvec, iv))
        if cfg.get("use_views", True):
            feat = np.concatenate(([dot], iv, [views.get(vid, 0.0)]), axis=0)
        else:
            feat = np.concatenate(([dot], iv), axis=0)
        rows.append((vid, feat.astype(np.float32)))

    if not rows:
        return []

    X = torch.from_numpy(np.vstack([r[1] for r in rows]))
    with torch.no_grad():
        scores = torch.sigmoid(model(X)).numpy().reshape(-1)
    out = [(vid, float(s)) for (vid, _), s in zip(rows, scores)]
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default=None, help="Which embedding model row to use (defaults to freshest)")
    ap.add_argument("--no-views", dest="no_views", action="store_true", help="Disable view-count side feature")
    ap.add_argument("--predict", type=str, default=None, help="Comma-separated video_ids to score")
    args = ap.parse_args()

    if args.predict:
        ids = [s.strip() for s in args.predict.split(",") if s.strip()]
        for vid, score in predict(ids):
            print(f"{vid}\t{score:.4f}")
    else:
        use_views = not args.no_views
        train(epochs=args.epochs, lr=args.lr, model_name=args.model, seed=args.seed, use_views=use_views)

if __name__ == "__main__":
    main()
