# src/cf_recs.py
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
    os.makedirs(CF_DIR, exist_ok=True)

def _load_history_for_cf(max_rows: int = 5000):
    # history rows: [(video_id, ts, dwell, disliked, skipped)] newest first
    rows = store.get_history(n=max_rows)
    return rows

def _build_interactions(rows) -> Optional[tuple]:
    if not rows:
        return None
    # Single-user CF: synth user "u0"
    user = "u0"
    vids = [vid for (vid, *_rest) in rows]
    items = list(dict.fromkeys(vids))  # stable unique
    if not items:
        return None
    # Positive if not explicitly skipped/disliked; you can tune later
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
    _ensure_dir()
    rows = _load_history_for_cf()
    built = _build_interactions(rows)
    if not built:
        return None, None
    user, items, ds, payload = built

    if not _HAS_LFM:
        # No LightFM installed; just persist item list so we can do a dummy rank
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
    # Returns (ids, scores) for the single user
    try:
        meta = load(DATASET_PATH)
        model_obj = load(MODEL_PATH)
    except Exception:
        return [], np.array([])
    item_ids: List[str] = meta["item_ids"]
    if not item_ids:
        return [], np.array([])

    if isinstance(model_obj, dict) and model_obj.get("dummy"):
        # no LightFM; return recent items as weak positives
        return item_ids[:n], np.linspace(1.0, 0.1, num=min(n, len(item_ids)))

    ds = meta["dataset"]
    inv_item_map = {v: k for k, v in ds._item_id_mapping.items()}
    all_item_indices = np.array(list(inv_item_map.keys()))
    # single user index is 0
    scores = model_obj.predict(0, all_item_indices)
    order = np.argsort(-scores)
    top = order[:min(n, len(order))]
    ids = [inv_item_map[int(i)] for i in all_item_indices[top]]
    return ids, scores[top]

if __name__ == "__main__":
    # allow: python -m src.cf_recs (trains)
    train_cf_pipeline()
    print("CF training done.")
