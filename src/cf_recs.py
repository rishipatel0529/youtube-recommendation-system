"""
cf_recs.py — Collaborative Filtering recommendations using LightFM.

- Builds single-user implicit feedback dataset from history
- Trains LightFM (if installed) or falls back to a dummy ranker
- Saves model/dataset via joblib
- Provides `recommend_cf()` to get top-N recommendations
"""

import os, itertools, numpy as np
from typing import List, Tuple, Dict, Optional
try:
    from lightfm import LightFM
    from lightfm.data import Dataset
    _HAS_LFM = True
except Exception:
    _HAS_LFM = False

from joblib import dump, load
from . import store

DATA_DIR = os.getenv("YT_REC_DATA_DIR", "data")
CF_DIR = os.path.join(DATA_DIR, "cf")
MODEL_PATH = os.path.join(CF_DIR, "lightfm_model.joblib")
DATASET_PATH = os.path.join(CF_DIR, "lightfm_dataset.joblib")

def _ensure_dir():
    # Ensure CF model directory exists.
    os.makedirs(CF_DIR, exist_ok=True)

def _load_history_for_cf(max_rows: int = 5000):
    # Fetch recent history rows for training CF model.
    rows = store.get_history(n=max_rows)
    return rows

def _build_interactions(rows) -> Optional[tuple]:
    # Convert history rows into LightFM interactions dataset.
    if not rows:
        return None
    user = "u0"
    vids = [vid for (vid, *_rest) in rows]
    items = list(dict.fromkeys(vids)) # unique order-preserving
    if not items:
        return None
    
    # Assign weights based on watch behavior
    weights: Dict[str, float] = {}
    for (vid, _ts, dwell, disliked, skipped) in rows:
        w = 1.0
        if skipped:   w -= 0.5
        if disliked:  w -= 1.0
        if dwell and dwell > 120: w += 0.25
        weights[vid] = max(0.0, w)

    if not _HAS_LFM:
        return ("u0", items, None, weights)

    ds = Dataset()
    ds.fit(users=[user], items=items)
    triples = ((user, vid, wt) for vid, wt in weights.items() if wt > 0)
    interactions, weights_arr = ds.build_interactions(triples)
    return (user, items, ds, (interactions, weights_arr))

def train_cf_pipeline():
    # Train CF model and persist model + dataset to disk.
    _ensure_dir()
    rows = _load_history_for_cf()
    built = _build_interactions(rows)
    if not built:
        return None, None
    user, items, ds, payload = built

    if not _HAS_LFM:
        dump({"user": user, "items": items, "dummy": True}, MODEL_PATH)
        dump({"dataset": None, "item_ids": items}, DATASET_PATH)
        return "dummy", items

    inter, w = payload
    model = LightFM(loss="warp")
    model.fit(inter, sample_weight=w, epochs=15, num_threads=4)
    dump(model, MODEL_PATH)
    dump({"dataset": ds, "item_ids": items}, DATASET_PATH)
    return model, ds

def recommend_cf(n: int = 200) -> Tuple[List[str], np.ndarray]:
    # Return top-N (ids, scores) for the single user.
    try:
        meta = load(DATASET_PATH)
        model_obj = load(MODEL_PATH)
    except Exception:
        return [], np.array([])
    item_ids: List[str] = meta["item_ids"]
    if not item_ids:
        return [], np.array([])

    if isinstance(model_obj, dict) and model_obj.get("dummy"):
        return item_ids[:n], np.linspace(1.0, 0.1, num=min(n, len(item_ids)))

    ds = meta["dataset"]
    inv_item_map = {v: k for k, v in ds._item_id_mapping.items()}
    all_item_indices = np.array(list(inv_item_map.keys()))
    scores = model_obj.predict(0, all_item_indices) # single user index = 0
    order = np.argsort(-scores)
    top = order[:min(n, len(order))]
    ids = [inv_item_map[int(i)] for i in all_item_indices[top]]
    return ids, scores[top]

if __name__ == "__main__":
    train_cf_pipeline()
    print("CF training done.")
