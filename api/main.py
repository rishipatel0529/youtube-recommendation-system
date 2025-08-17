from fastapi import FastAPI, Query
from .schemas import (
    RecRequest, SearchRequest, InteractRequest, RecItem,
    BlockRequest, SaveVideoRequest, SaveChannelRequest,
    SubscribeRequest, UnsubscribeRequest, TrendRequest, CategoryRequest
)
from . import service
import time, datetime as dt

# API layer: thin FastAPI routes that validate inputs, call service, and return clean JSON.
app = FastAPI(title="YT Recommender API")

@app.get("/healthz")
def healthz():
    # Liveness/readiness check.
    return {"ok": True}

@app.post("/recommend", response_model=list[RecItem])
def recommend(req: RecRequest):
    # Return one page of ranked recommendations and log impressions.
    items = service.recommend_via_pager(req.user_id, req.page, req.page_size, req.variant)
    for it in items:
        service.log_interaction({
            "user_id": req.user_id, "video_id": it["video_id"], "rank": it["rank"],
            "variant": req.variant, "event": "impression", "page": req.page, "is_search": 0
        })
    return items


@app.post("/search", response_model=list[RecItem])
def search(req: SearchRequest):
    # Return one page of search results (pager-integrated).
    items = service.search_via_pager(req.q, req.page, req.page_size)
    return items


@app.post("/interact")
def interact(req: InteractRequest):
    # Log a user interaction event (watch/like/skip/etc.).
    service.log_interaction(req.dict())
    return {"ok": True}


@app.get("/ab/report")
def ab_report(since: str = Query(..., description="YYYY-MM-DD")):
    # Aggregate AB events since a given date (UTC midnight).
    import datetime as dt
    ts = int(dt.datetime.fromisoformat(since).timestamp())
    return service.ab_report_since(ts)

@app.post("/action/block")
def action_block(req: BlockRequest):
    # Block a channel by id (persists to DB/CSV).
    return service.block_channel(req.channel_id, req.channel_title)

@app.post("/action/save_video")
def action_save_video(req: SaveVideoRequest):
    # Save a video to local favorites.
    return service.save_video(req.video_id)

@app.post("/action/save_channel")
def action_save_channel(req: SaveChannelRequest):
    # Save a channel to local favorites.
    return service.save_channel(req.channel_id, req.channel_title)

@app.get("/subs")
def get_subs():
    # List locally stored subscriptions.
    return {"items": service.list_subscriptions()}

@app.post("/subs/subscribe")
def subs_subscribe(req: SubscribeRequest):
    # Add a local subscription record.
    return service.subscribe(req.channel_id, req.channel_title, req.channel_url)

@app.post("/subs/unsubscribe")
def subs_unsubscribe(req: UnsubscribeRequest):
    # Remove a local subscription record.
    return service.unsubscribe(req.channel_id)

@app.get("/categories")
def categories_list():
    # Return top categories and counts for filtering.
    return {"items": service.list_categories()}

@app.post("/control/trending")
def control_trending(req: TrendRequest):
    # Toggle trending-only mode.
    return service.set_trending_only(req.on)

@app.post("/control/category")
def control_category(req: CategoryRequest):
    # Set or clear the active category filter.
    return service.set_active_category(req.category_id)

@app.post("/control/refresh")
def control_refresh():
    # Clear server-side page cache.
    service.clear_pagecache()
    return {"ok": True}
