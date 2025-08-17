# YouTube Terminal Recommender (Content-Based, Session-Learning)

A terminal app that starts with 10 popular videos and then adapts to your selections,
recommending similar content while boosting creators you've engaged with.

## Features
- Starts from YouTube's most popular videos for a cold start
- Learns from your selections in-session and across runs
- Content-based TF‑IDF similarity over titles, descriptions, and tags
- Boosts same‑channel and recently watched creators
- Expands the candidate pool with YouTube's `relatedToVideoId`
- Filters out Shorts by duration (<60s) and common tags
- Local SQLite cache of videos and your watch history

## Setup
1) Create a `.env` with your API key:
```
YT_API_KEY=YOUR_API_KEY_HERE
REGION=US
```
2) Install deps:
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
3) Run:
```
python -m src.main
```

## Controls
- Enter a number to "watch" that video and update recommendations
- `r` to refresh recommendations
- `h` to see watch history
- `q` to quit

## Notes
- Uses only public YouTube Data API v3 endpoints via `requests`
- DB file: `data/app.db`
- Cached thumbnails/extra fields are not stored; this is text‑only terminal app
