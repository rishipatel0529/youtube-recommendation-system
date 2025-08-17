"""
feats.py — Text and metadata utilities for video processing.

Purpose:
- Extract hashtags from video metadata.
- Detect language of text safely.
- Bucket video durations into categories.
- Compute similarity scores (Jaccard).
- Optionally score recency for ranking.
"""

import re, math, datetime
from langdetect import detect, LangDetectException

_HASHTAG = re.compile(r"(?:#|\b)([A-Za-z0-9_]{2,})")

def extract_hashtags(title, desc, tags):
    # Extract hashtags from title, description, and tags; filter noise.
    s = " ".join([title or "", desc or "", " ".join(tags or [])])
    cand = set(x.lower() for x in _HASHTAG.findall(s))
    return {h for h in cand if h not in {"shorts", "short"}}

def detect_language(text):
    # Detect language of text; return 'unknown' if detection fails.
    try:
        return detect(text) if text and text.strip() else "unknown"
    except LangDetectException:
        return "unknown"

def duration_bucket(sec):
    # Categorize duration (sec) into 'short', 'medium', or 'long'.
    if sec is None: return "unknown"
    s = int(sec)
    if s < 300: return "short"
    if s <= 1200: return "medium"
    return "long"

def jaccard(a:set, b:set):
    # Compute Jaccard similarity between two sets.
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

def recency_score(published_at):
    # Optional: Score recency (higher = newer); unused in current ranker.
    try:
        dt = datetime.datetime.fromisoformat(published_at.replace("Z","+00:00"))
    except Exception:
        return 0.0
    days = max((datetime.datetime.now(datetime.timezone.utc) - dt).days, 0)
    return 1.0 / (1.0 + math.log1p(days))
