"""
# src/embeds_faiss.py
import os, json, time
import numpy as np
import faiss

try:
    from fastembed import TextEmbedding
except Exception as e:
    TextEmbedding = None

DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH  = os.path.join(DATA_DIR, "faiss.meta.json")

_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, lightweight
_EMBEDDER = None
_INDEX = None
_IDS = []  # row -> video_id
_DOC_HASH = ""  # to detect store changes

def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        if TextEmbedding is None:
            raise RuntimeError("fastembed is not installed")
        _EMBEDDER = TextEmbedding(model_name=_MODEL_NAME)
    return _EMBEDDER

def _embed(texts):
    emb = list(_get_embedder().embed(texts))
    X = np.array(emb, dtype=np.float32)
    # guard against zero rows
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    return X / norms

def _video_to_text(v):
    # what you index: title + (optional) description/tags
    t = (v.get("title") or "").strip()
    d = (v.get("description") or "").strip()
    return f"{t}\n\n{d}" if d else t

def _hash_for_videos(videos):
    # cheap content hash (count + last modified-ish)
    return f"{len(videos)}:{sum(len((v.get('title') or '')) for v in videos)%7919}"

def _save(index, ids):
    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump({
            "model": _MODEL_NAME,
            "ids": ids,
            "ts": int(time.time()),
        }, f)

def _load():
    global _INDEX, _IDS
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return False
    _INDEX = faiss.read_index(INDEX_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    _IDS = meta["ids"]
    return True

def rebuild(videos):
    global _INDEX, _IDS, _DOC_HASH
    texts = [_video_to_text(v) for v in videos]
    ids = [v["id"] for v in videos if v.get("id")]
    X = _embed(texts)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on normalized vectors == cosine
    index.add(X)
    _INDEX = index
    _IDS = ids
    _DOC_HASH = _hash_for_videos(videos)
    _save(index, ids)

def ensure_ready(videos):
    global _DOC_HASH
    want = _hash_for_videos(videos)
    if _INDEX is None:
        if not _load():
            rebuild(videos)
            return
    # stale?
    if _DOC_HASH != want:
        rebuild(videos)

def is_ready():
    return _INDEX is not None and len(_IDS) > 0

def search(query_texts, topn=200):
    if not is_ready():
        return []
    Q = _embed(query_texts)
    # average queries into a single centroid (simple & robust)
    q = np.mean(Q, axis=0, keepdims=True)
    D, I = _INDEX.search(q, topn)
    scores = D[0].tolist()
    idxs = I[0].tolist()
    out = []
    for i, s in zip(idxs, scores):
        if i < 0 or i >= len(_IDS):
            continue
        out.append((_IDS[i], float(s)))
    return out
"""