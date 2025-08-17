"""
recommender.py — Content-based + hybrid ranking for YouTube-style recommendations.

- Builds TF-IDF vectors and metadata features from cached videos
- Creates a recency-weighted user profile from watch history
- Ranks with either learned linear model (ranker) or a manual fallback
- Optional Phase-3 diversity via MMR and hybrid blending with CF (LightFM)
"""

import numpy as np
import os, time, datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from . import ranker
except ImportError:
    import ranker # fallback

from .feats import extract_hashtags, detect_language, duration_bucket, jaccard

HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5")) # 0..1, higher favors content
USE_CF = os.getenv("USE_CF", "lightfm").lower()
MIN_LONG_SEC = int(os.getenv("LONG_MIN_SEC", "180"))
RECENT_BOOST_N = int(os.getenv("RECENT_BOOST_N", "10"))
RECENT_BOOST_FACTOR = float(os.getenv("RECENT_BOOST_FACTOR", "2.0"))
USE_SUBS = os.getenv("USE_SUBS", "1").lower() in {"1","true","yes"}
SUBS_MAX = int(os.getenv("SUBS_MAX", "300"))

# Phase 3 knobs
DIVERSITY_STRENGTH = float(os.getenv("DIVERSITY_STRENGTH", "0.25")) # 0..1 (MMR lambda)
FRESH_HALF_LIFE_DAYS = float(os.getenv("FRESH_BOOST_HALF_LIFE_DAYS", "14"))
CREATOR_COOLDOWN = float(os.getenv("CREATOR_COOLDOWN", "0.35")) # penalty for repeats

def _looks_short_title(t: str) -> bool:
    # Heuristic: title contains #short/#shorts/‘ shorts ’.
    t = (t or "").lower()
    return ("#short" in t) or ("#shorts" in t) or (" shorts " in t)

def _is_longform_from_model(model, idx: int) -> bool:
    # Return True if item at idx looks long-form given duration/title.
    dur = int(model["dur"][idx] or 0)
    title = model["titles"][idx] if "titles" in model else ""
    return (dur >= MIN_LONG_SEC) and (not _looks_short_title(title))

def _to_epoch(s):
    # Parse ISO8601 string to epoch seconds, else 0.
    if not s: return 0
    try:
        dt = datetime.datetime.fromisoformat(s.replace("Z","+00:00"))
        return int(dt.timestamp())
    except Exception:
        return 0

def _freshness_score(published_at_iso, half_life_days=14.0, now_ts=None):
    # Exponential-decay freshness in [0,1]: 1 at publish time, 0.5 at half-life.
    now = int(now_ts or time.time())
    ts = _to_epoch(published_at_iso)
    if ts <= 0:
        return 0.0
    age_days = max(0.0, (now - ts) / 86400.0)
    return float(2.0 ** (-age_days / max(0.1, half_life_days)))

def _age_days(published_at: str) -> int:
    # Age in whole days from ISO8601, 0 on error.
    try:
        dt = datetime.datetime.fromisoformat((published_at or "").replace("Z","+00:00"))
        return max(0, int((datetime.datetime.now(datetime.timezone.utc) - dt).days))
    except Exception:
        return 0

def build_vectors(videos):
    # Vectorize titles+descriptions (TF-IDF) and collect per-video metadata.
    texts, ids, channels, views = [], [], [], []
    titles, descs, tags_list = [], [], []
    durs, langs, hashtags, cat_ids, pub = [], [], [], [], []
    for v in videos:
        tags = [t for t in (v.get("tags") or []) if isinstance(t, str)]
        txt = " ".join([v.get("title",""), v.get("description",""), " ".join(tags)])
        texts.append(txt); ids.append(v["id"]); channels.append(v.get("channel_id",""))
        views.append(int(v.get("view_count") or 0))
        titles.append(v.get("title","")); descs.append(v.get("description","")); tags_list.append(tags)
        durs.append(int(v.get("duration_sec") or 0))
        langs.append(detect_language(v.get("title","") + " " + v.get("description","")))
        hashtags.append(extract_hashtags(v.get("title",""), v.get("description",""), tags))
        cat_ids.append(v.get("category_id") or "")
        pub.append(v.get("published_at") or "")
    if not texts:
        return None
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words="english")
    X = vec.fit_transform(texts)
    views = np.array(views, dtype=float)
    pop = (np.log1p(views) - np.log1p(views).min()) / (np.ptp(np.log1p(views)) + 1e-9) if views.size else np.zeros_like(views)
    return {
        "vectorizer": vec, "X": X, "ids": ids, "channels": channels, "pop": pop,
        "titles": titles, "descs": descs, "tags": tags_list, "dur": durs, "lang": langs,
        "hashtags": hashtags, "cat": cat_ids, "published": pub
    }

def make_user_profile(model, videos, history_rows, decay=0.9):
    # Build a recency-weighted sparse user vector from watch history.
    if not history_rows:
        return None
    id_to_idx = {vid: i for i, vid in enumerate(model["ids"])}
    weights, vecs = [], []

    # Iterate oldest→newest so power grows toward most recent
    seq = history_rows[::-1]
    power = 1.0
    for j, (vid, _ts, dwell, disliked, skipped) in enumerate(seq):
        i = id_to_idx.get(vid)
        if i is None:
            continue
        w = power
        if disliked or skipped:
            w *= -0.5
        # Recency boost for latest N watches
        if j >= len(seq) - RECENT_BOOST_N:
            w *= RECENT_BOOST_FACTOR
        vecs.append(model["X"][i])
        weights.append(w)
        power *= decay

    if not vecs:
        return None
    W = np.array(weights, dtype=float)
    W = W / (np.sum(np.abs(W)) + 1e-9)
    V = vecs[0].multiply(W[0])
    for j in range(1, len(vecs)):
        V = V + vecs[j].multiply(W[j])
    return V

def _stage1_topk(model, user_vec, last_vid, k=200):
    # Retrieve a candidate pool by TF-IDF similarity (+blend with last video).
    ids = model["ids"]
    sims = cosine_similarity(user_vec, model["X"]).ravel() if user_vec is not None else np.zeros(len(ids))
    if last_vid is not None and last_vid in ids:
        li = ids.index(last_vid)
        sims = 0.7*sims + 0.3*cosine_similarity(model["X"][li], model["X"]).ravel()
    order = np.argsort(-sims)
    return order[:min(k, len(order))], sims

def _hist_prefs(model, history_rows, all_videos_map):
    # Summarize history preferences: top language, duration bucket, hashtags, channel stats.
    if not history_rows:
        sub_set = set()
        if USE_SUBS:
            try:
                from . import store
                sub_set = set(list(store.get_subscribed_channel_ids())[:SUBS_MAX])
            except Exception:
                sub_set = set()
        return {"lang": "unknown", "channel_boost": sub_set, "chan_success": {}}, "unknown", set()

    langs, chans, dur_buckets, hash_all = [], [], [], set()
    stats = {}
    for vid, _ts, dwell, disliked, skipped in history_rows:
        v = all_videos_map.get(vid)
        if not v: continue
        lang = model["lang"][model["ids"].index(vid)] if vid in model["ids"] else "unknown"
        langs.append(lang)
        c = v.get("channel_id","")
        chans.append(c)
        dur_buckets.append(duration_bucket(v.get("duration_sec",0)))
        hash_all |= extract_hashtags(v.get("title",""), v.get("description",""), v.get("tags"))
        d = stats.setdefault(c, {"n":0,"likes":0})
        d["n"] += 1
        if not (disliked or skipped):
            d["likes"] += 1

    from collections import Counter
    lang_top = Counter(langs).most_common(1)[0][0] if langs else "unknown"
    dur_top  = Counter(dur_buckets).most_common(1)[0][0] if dur_buckets else "unknown"
    top_chans = set([c for c,_ in Counter(chans).most_common(5)])

    sub_set = set()
    if USE_SUBS:
        try:
            from . import store
            sub_set = set(list(store.get_subscribed_channel_ids())[:SUBS_MAX])
        except Exception:
            sub_set = set()

    boost = top_chans | sub_set
    success = {}
    for c, d in stats.items():
        success[c] = (d["likes"] / d["n"]) if d["n"] else 0.0

    return {"lang": lang_top, "channel_boost": boost, "chan_success": success}, dur_top, hash_all

def _feature_row(model, idx, user_vec, last_vec, hist_lang_pref, hist_dur_pref, hist_hashtags,
                 session_recent_channels=None, session_topic_freqs=None, now_ts=None):
    # Assemble per-item features used by the ranker (manual or learned).
    ids = model["ids"]; channels = model["channels"]; pop = model["pop"]
    dur = model["dur"][idx]; lang = model["lang"][idx]; tags = model["hashtags"][idx]
    sim_user = cosine_similarity(user_vec, model["X"][idx]).ravel()[0] if user_vec is not None else 0.0
    sim_last = cosine_similarity(last_vec, model["X"][idx]).ravel()[0] if last_vec is not None else 0.0
    same_channel = 1.0 if hist_lang_pref.get("channel_boost", set()) and channels[idx] in hist_lang_pref["channel_boost"] else 0.0
    dur_match = 1.0 if duration_bucket(dur) == hist_dur_pref else 0.0
    lang_match = 1.0 if (lang != "unknown" and lang == hist_lang_pref.get("lang")) else 0.0
    hashtag_overlap = jaccard(tags, hist_hashtags)

    # freshness
    fresh = _freshness_score(model["published"][idx], half_life_days=FRESH_HALF_LIFE_DAYS, now_ts=now_ts)

    # session penalties/bonuses
    cooldown = 1.0 if (session_recent_channels and channels[idx] in session_recent_channels) else 0.0
    seen_success = hist_lang_pref.get("chan_success", {}).get(channels[idx], None)
    novelty = 1.0 if seen_success is None else 0.0
    fatigue = 0.0

    if session_topic_freqs and tags:
        hits = sum(session_topic_freqs.get(t, 0) for t in tags)
        fatigue = 1.0 if hits >= 2 else (0.5 if hits == 1 else 0.0)

    return {
        "sim_user": float(sim_user),
        "sim_last": float(sim_last),
        "pop": float(pop[idx]),
        "same_channel": float(same_channel),
        "dur_match": float(dur_match),
        "lang_match": float(lang_match),
        "hashtag_overlap": float(hashtag_overlap),
        "fresh": float(fresh),
        "cooldown": float(cooldown),
        "novelty": float(novelty),
        "fatigue": float(fatigue),
        "channel": channels[idx],
        "published": model["published"][idx],
    }

def _manual_score(fv):
    # Hand-tuned linear weights; cooldown/fatigue as negative terms.
    W = {
        "sim_user": 0.50,
        "sim_last": 0.15,
        "pop": 0.08,
        "same_channel": 0.04,
        "dur_match": 0.03,
        "lang_match": 0.03,
        "hashtag_overlap": 0.05,
        "fresh": 0.06,
        "novelty": 0.04,
        "cooldown": -CREATOR_COOLDOWN, # penalty
        "fatigue": -0.10, # penalty
    }
    s = sum(W[k] * fv[k] for k in W.keys())
    return s, W

def explain_reasons(fv, subs_set=None, max_reasons=3):
    # Produce short user-facing reason strings from a feature row.
    r = []
    if fv.get("sim_user", 0) > 0.35: r.append("topic match")
    if fv.get("hashtag_overlap", 0) >= 0.2: r.append("shared hashtags")
    if fv.get("fresh", 0) >= 0.5: r.append("fresh")
    if subs_set and fv.get("channel") in subs_set: r.append("from a subscription")
    if fv.get("novelty", 0) >= 0.9: r.append("new channel for you")
    if fv.get("cooldown", 0) >= 0.9: r.append("recently seen this creator (deprioritized)")
    if fv.get("fatigue", 0) >= 0.9: r.append("you’ve seen a lot on this topic (deprioritized)")
    return r[:max_reasons]

def _mmr_select(model, idx_list, scores, k=20, lam=0.75):
    # Maximal Marginal Relevance selection to promote diversity among top items.
    if not idx_list: return []
    selected = []
    X = model["X"]
    ids = idx_list[:]
    sims_cache = {}

    def item_sim(i, j):
        key = (i, j) if i <= j else (j, i)
        if key in sims_cache: return sims_cache[key]
        s = cosine_similarity(X[i], X[j]).ravel()[0]
        sims_cache[key] = float(s)
        return s

    S = []
    while len(S) < min(k, len(ids)):
        best_i, best_val = None, -1e9
        for pos, idx in enumerate(ids):
            if idx in S: continue
            rel = scores[pos]
            div = 0.0
            if S:
                div = max(item_sim(idx, j) for j in S)
            val = lam * rel - (1 - lam) * div
            if val > best_val:
                best_i, best_val = pos, val
        if best_i is None:
            break
        # keep alignment by removing chosen element
        S.append(ids[best_i])
        del ids[best_i]
        del scores[best_i]
    return S

def rank(model, user_vec, exclude_ids, k=20, last_video=None, history_rows=None, all_videos=None,
         session_recent_channels=None, session_topic_freqs=None, use_phase3=True, return_contribs=False):
    # Primary content-based ranking pipeline with optional diversity and contributions.
    # freeze page size to avoid shadowing
    try:
        k_int = int(k)
    except Exception:
        k_int = 20

    # Stage 1: get a big pool by TF-IDF similarity
    idxs, sims = _stage1_topk(model, user_vec, last_video, k=500)
    ids = model["ids"]

    # Build allowed set (exclude, enforce long-form)
    exclude_ids = exclude_ids or set()
    allowed = []
    for idx in idxs:
        vid = ids[idx]
        if vid in exclude_ids:
            continue
        if not _is_longform_from_model(model, idx):
            continue
        allowed.append(idx)
    if not allowed:
        return ([], {}) if return_contribs else []

    # Last vec
    lst_vec = model["X"][ids.index(last_video)] if (last_video in ids) else None

    # History prefs
    rnk = ranker.load_model()
    hist_lang_pref, hist_dur_pref, hist_hash = _hist_prefs(
        model, history_rows or [], {v["id"]: v for v in (all_videos or [])}
    )

    # Build features for candidate set
    feats = []
    keep = [] # (vid, fv, idx)
    now_ts = int(time.time())
    for idx in allowed:
        vid = ids[idx]
        fv = _feature_row(
            model, idx, user_vec, lst_vec, hist_lang_pref, hist_dur_pref, hist_hash,
            session_recent_channels=session_recent_channels,
            session_topic_freqs=session_topic_freqs,
            now_ts=now_ts
        )
        feats.append(ranker.vectorize_feature_row({key: val for key, val in fv.items() if isinstance(val, (int,float))}))
        keep.append((vid, fv, idx))

    if not keep:
        return ([], {}) if return_contribs else []

    X2 = np.vstack(feats)

    # Choose scores: learned linear vs manual fallback
    use_manual = False
    try:
        coef = getattr(rnk, "coef_", None)
        if coef is None or np.allclose(coef, 0):
            use_manual = True
    except Exception:
        use_manual = True

    if use_manual:
        scores = []
        weights = None
        for _vid, fv, _idx in keep:
            s, weights = _manual_score(fv)
            # gentle age penalty for very old uploads
            age = _age_days(fv.get("published",""))
            age_pen = -0.03 if age > 365 else 0.0
            s += age_pen
            fv["_age_penalty_cache"] = float(age_pen)
            scores.append(s)
    else:
        weights = None
        scores = ranker.predict_scores(rnk, X2)

    # Phase-3: apply MMR diversity on top chunk
    order = list(np.argsort(-scores))
    if use_phase3:
        topN = min(50, len(order))
        idx_list = [keep[j][2] for j in order[:topN]]
        rel = [float(scores[j]) for j in order[:topN]]
        mmr_idxs = _mmr_select(model, idx_list, rel, k=topN, lam=1.0 - DIVERSITY_STRENGTH)
        mmr_set = set(mmr_idxs)
        order = [j for j in order if keep[j][2] in mmr_set] + [j for j in order if keep[j][2] not in mmr_set]

    out = []
    contribs = {}
    for j in order:
        vid, fv, _idx = keep[j]
        if vid in exclude_ids:
            continue
        out.append((vid, float(scores[j])))
        if return_contribs:
            if weights is not None:
                comp = {}
                sc, W = _manual_score(fv)
                for feat_name, wgt in W.items():
                    comp[feat_name] = float(wgt * fv.get(feat_name, 0.0))
                age_pen = float(fv.get("_age_penalty_cache", 0.0))
                comp["age_penalty"] = age_pen
                comp["_total"] = float(sc + age_pen)
            else:
                comp = {key: float(val) for key, val in fv.items() if isinstance(val, (int,float))}
                comp["_total"] = float(scores[j])
            contribs[vid] = comp
        if len(out) >= k_int:
            break

    # Backfill popular long-form if needed
    if len(out) < k_int and all_videos is not None:
        watched = exclude_ids
        popular_pool = sorted(all_videos, key=lambda x: x.get("view_count", 0), reverse=True)
        seen = {vid for vid, _ in out} | watched
        for v in popular_pool:
            if v["id"] in seen:
                continue
            if int(v.get("duration_sec") or 0) < MIN_LONG_SEC or _looks_short_title(v.get("title")):
                continue
            out.append((v["id"], 0.0))
            if len(out) >= k_int:
                break

    return (out[:k_int], contribs) if return_contribs else out[:k_int]

def rank_hybrid(model, user_vec, exclude_ids, k=20, last_video=None, history_rows=None, all_videos=None,
                session_recent_channels=None, session_topic_freqs=None, use_phase3=True, return_contribs=False):
    
    # Blend content-based scores with CF scores (LightFM) via HYBRID_ALPHA.
    # freeze page size
    try:
        k_int = int(k)
    except Exception:
        k_int = 20

    # 1) Content
    content, contribs = rank(
        model, user_vec, exclude_ids, k=max(200, k_int), last_video=last_video,
        history_rows=history_rows, all_videos=all_videos,
        session_recent_channels=session_recent_channels,
        session_topic_freqs=session_topic_freqs,
        use_phase3=use_phase3, return_contribs=True
    )
    cb_scores = {vid: s for vid, s in content}

    # 2) CF (optional)
    cf_ids, cf_scores = [], np.array([])
    try:
        if USE_CF == "lightfm":
            from .cf_recs import recommend_cf
            cf_ids, cf_scores = recommend_cf(n=400)
    except Exception:
        pass
    allowed = set(v["id"] for v in (all_videos or [])) - set(exclude_ids or set())
    cf_pairs = [(vid, float(sc)) for vid, sc in zip(cf_ids, cf_scores) if vid in allowed]

    # 3) Blend (normalize CF band to [0,1] then mix)
    blended = dict(cb_scores)
    if cf_pairs:
        cf_vals = np.array([sc for _vid, sc in cf_pairs], dtype=float)
        if cf_vals.size:
            cf_norm = (cf_vals - cf_vals.min()) / (cf_vals.ptp() + 1e-9)
            for (vid, _), scn in zip(cf_pairs, cf_norm):
                blended[vid] = blended.get(vid, 0.0) * HYBRID_ALPHA + (1.0 - HYBRID_ALPHA) * scn

    # 4) Final order
    ordered = sorted(blended.items(), key=lambda x: -x[1])
    out = []
    seen = set()
    for vid, sc in ordered:
        if vid in seen:
            continue
        seen.add(vid)
        out.append((vid, sc))
        if len(out) >= k_int:
            break

    return (out[:k_int], contribs) if return_contribs else out[:k_int]
