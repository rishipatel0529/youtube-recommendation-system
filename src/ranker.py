"""
ranker.py - Training and persistence logic for the recommendation ranker.

- Defines feature vectorization and model persistence (save/load).
- Provides prediction functions using linear models.
- Implements full retraining and partial-fit updates from DB history.
- Includes basic evaluation metrics (AUC, MAP@K).
"""

import os, json, time, pathlib
import numpy as np

# Config thresholds from env (with defaults)
MIN_TOTAL   = int(os.getenv("PF_MIN_TOTAL", "80")) # minimum samples per batch
MIN_POS     = int(os.getenv("PF_MIN_POS",   "5")) # minimum positives required
MIN_NEG     = int(os.getenv("PF_MIN_NEG",   "5")) # minimum negatives required
WINDOW_K    = int(os.getenv("PF_WINDOW_K",  "200")) # sliding window size

try:
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import roc_auc_score
except Exception:
    # Fallback if sklearn not installed
    SGDRegressor = None
    def roc_auc_score(y_true, y_score): return 0.5

# File paths
DATA_DIR = pathlib.Path(os.getenv("YT_REC_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = DATA_DIR / "ranker.pkl"

# Feature schema
FEATURE_FIELDS = ["sim_user","sim_last","pop","same_channel","dur_match","lang_match","hashtag_overlap","recency"]

# Feature handling
def vectorize_feature_row(fv: dict):
    # Convert feature dict into numeric vector in fixed field order.
    return np.array([float(fv.get(k, 0.0)) for k in FEATURE_FIELDS], dtype=float)

# Model persistence

def save_model(model, path=None):
    # Save model to disk (joblib if available, else numpy fallback).
    path = pathlib.Path(path or MODEL_PATH)
    try:
        import joblib
        joblib.dump(model, path)
    except Exception:
        arr = getattr(model, "coef_", np.zeros(len(FEATURE_FIELDS)))
        with open(path, "wb") as f:
            np.save(f, arr)

def load_model(path=None):
    # Load model from disk or return stub with zero weights.
    path = pathlib.Path(path or MODEL_PATH)
    if not path.exists():
        class Stub:
            coef_ = np.zeros(len(FEATURE_FIELDS))
            intercept_ = 0.0
        return Stub()
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        coef = np.load(path)
        class Simple:
            def __init__(self, coef):
                self.coef_ = coef
                self.intercept_ = 0.0
            def predict(self, X):
                return X.dot(self.coef_) + self.intercept_
        return Simple(coef)

# Prediction
def predict_scores(model, X: np.ndarray):
    # Return prediction scores for feature matrix X.
    if hasattr(model, "predict"):
        return model.predict(X).ravel()
    coef = getattr(model, "coef_", np.zeros(X.shape[1]))
    b = getattr(model, "intercept_", 0.0)
    return (X.dot(coef) + b).ravel()

# Training helpers

def _split_holdout(idx_list, holdout_frac=0.2, seed=42):
    # Train/test split with random shuffle.
    rng = np.random.RandomState(seed)
    idx = np.arange(len(idx_list))
    rng.shuffle(idx)
    cut = int(len(idx) * (1.0 - holdout_frac))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    return [idx_list[i] for i in train_idx], [idx_list[i] for i in test_idx]

def _map_at_k(y_true, scores, k=10):
    # Compute MAP@K metric from binary labels and scores.
    order = np.argsort(-scores)
    rel = np.array(y_true)[order][:k]
    if rel.sum() == 0:
        return 0.0
    precs = []
    hits = 0
    for i, r in enumerate(rel, start=1):
        if r == 1:
            hits += 1
            precs.append(hits / i)
    return float(np.mean(precs)) if precs else 0.0

def _build_training_matrix(model, all_videos, history_rows):
    """
    Build X, y from history:
      - y=1: liked / not disliked
      - y=0: disliked (plus sampled negatives)
    """
    from .recommender import _feature_row, _hist_prefs, make_user_profile

    id2v   = {v["id"]: v for v in all_videos}
    id2idx = {vid: i for i, vid in enumerate(model["ids"])}

    label_by_vid = {}
    for vid, ts, dwell, disliked, skipped in history_rows:
        if skipped:
            continue
        if vid in label_by_vid:
            continue
        label_by_vid[vid] = 0 if int(disliked) else 1

    pos_ids = [vid for vid, lab in label_by_vid.items() if lab == 1]
    neg_ids = [vid for vid, lab in label_by_vid.items() if lab == 0]

    watched = set(label_by_vid.keys())
    pool = [v for v in all_videos if v["id"] not in watched]
    rng = np.random.RandomState(123)
    extra_negs = [v["id"] for v in rng.choice(
        pool, size=min(len(pos_ids) * 3, len(pool)), replace=False
    )] if pool else []
    neg_ids.extend(extra_negs)

    hist_lang_pref, hist_dur_pref, hist_hash = _hist_prefs(model, history_rows or [], id2v)

    last_video = history_rows[0][0] if history_rows else None
    last_vec = model["X"][id2idx[last_video]] if last_video in id2idx else None
    user_vec = make_user_profile(model, all_videos, history_rows) if history_rows else None

    X_rows, y = [], []
    for vid in pos_ids:
        idx = id2idx.get(vid)
        if idx is None: continue
        fv = _feature_row(model, idx, user_vec, last_vec, hist_lang_pref, hist_dur_pref, hist_hash)
        X_rows.append(vectorize_feature_row(fv)); y.append(1)
    for vid in neg_ids:
        idx = id2idx.get(vid)
        if idx is None: continue
        fv = _feature_row(model, idx, user_vec, last_vec, hist_lang_pref, hist_dur_pref, hist_hash)
        X_rows.append(vectorize_feature_row(fv)); y.append(0)

    if not X_rows:
        return None, None
    return np.vstack(X_rows).astype(float), np.asarray(y, dtype=int)

# Training entrypoints

def retrain_from_db(save_path=None):
    # Full offline retrain using all history (with holdout eval).
    from . import store, recommender
    all_videos = store.fetch_all_videos()
    hist = store.get_history(n=5000)
    if not all_videos or not hist or SGDRegressor is None:
        return {"auc": 0.5, "map10": 0.0}

    model = recommender.build_vectors(all_videos)
    X, y = _build_training_matrix(model, all_videos, hist)
    if X is None:
        return {"auc": 0.5, "map10": 0.0}

    idx = list(range(len(y)))
    tr_idx, te_idx = _split_holdout(idx, holdout_frac=0.2)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    m = SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
    m.fit(Xtr, ytr)

    scores = m.predict(Xte)
    try:
        auc = float(roc_auc_score(yte, scores))
    except Exception:
        auc = 0.5
    map10 = _map_at_k(yte, scores, k=10)

    if save_path:
        save_model(m, save_path)
    return {"auc": auc, "map10": float(map10)}

def partial_fit_from_db(save_path=None):
    # Online update with sliding window over recent history.
    from . import store, recommender
    all_videos = store.fetch_all_videos()
    hist = store.get_history(n=2000)
    if not all_videos or not hist or SGDRegressor is None:
        return {"auc": 0.5, "map10": 0.0}

    model = recommender.build_vectors(all_videos)
    X, y = _build_training_matrix(model, all_videos, hist)
    if X is None:
        return {"auc": 0.5, "map10": 0.0}

    start = max(0, len(y) - WINDOW_K)
    if start > 0:
        X = X[start:]
        y = y[start:]

    total = len(y)
    if total < MIN_TOTAL:
        print(f"[partial-fit] skipped: only {total} examples (<{MIN_TOTAL})")
        return {"auc": float("nan"), "map10": 0.0}

    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos < MIN_POS or neg < MIN_NEG:
        print(f"[partial-fit] skipped: pos={pos}, neg={neg} (need ≥{MIN_POS}/≥{MIN_NEG})")
        return {"auc": float("nan"), "map10": 0.0}

    print(f"[partial-fit] batch total={total} pos={pos} neg={neg}")

    m = load_model()
    if m is None or not hasattr(m, "partial_fit"):
        m = SGDRegressor(random_state=42, max_iter=200, tol=1e-3)
        m.fit(X, y)
    else:
        m.partial_fit(X, y)

    tail = min(400, len(y))
    Xte, yte = X[-tail:], y[-tail:]
    scores = predict_scores(m, Xte)
    try:
        auc = float(roc_auc_score(yte, scores))
    except Exception:
        auc = 0.5
    map10 = _map_at_k(yte, scores, k=10)

    if save_path:
        save_model(m, save_path)
    return {"auc": auc, "map10": float(map10)}
