from fastapi import FastAPI, Query
from .schemas import (
    RecRequest, SearchRequest, InteractRequest, RecItem,
    BlockRequest, SaveVideoRequest, SaveChannelRequest,
    SubscribeRequest, UnsubscribeRequest, TrendRequest, CategoryRequest
)
from . import service
import time, datetime as dt


app = FastAPI(title="YT Recommender API")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/recommend", response_model=list[RecItem])
def recommend(req: RecRequest):
    items = service.recommend_via_pager(req.user_id, req.page, req.page_size, req.variant)
    # log impressions only for recommendations
    for it in items:
        service.log_interaction({
            "user_id": req.user_id, "video_id": it["video_id"], "rank": it["rank"],
            "variant": req.variant, "event": "impression", "page": req.page, "is_search": 0
        })
    return items


@app.post("/search", response_model=list[RecItem])
def search(req: SearchRequest):
    items = service.search_via_pager(req.q, req.page, req.page_size)
    # do NOT log impressions for search results (we want them to persist)
    return items


@app.post("/interact")
def interact(req: InteractRequest):
    service.log_interaction(req.dict())
    return {"ok": True}


@app.get("/ab/report")
def ab_report(since: str = Query(..., description="YYYY-MM-DD")):
    import datetime as dt
    ts = int(dt.datetime.fromisoformat(since).timestamp())
    return service.ab_report_since(ts)

# ---- NEW: actions ----

@app.post("/action/block")
def action_block(req: BlockRequest):
    return service.block_channel(req.channel_id, req.channel_title)

@app.post("/action/save_video")
def action_save_video(req: SaveVideoRequest):
    return service.save_video(req.video_id)

@app.post("/action/save_channel")
def action_save_channel(req: SaveChannelRequest):
    return service.save_channel(req.channel_id, req.channel_title)

# ---- NEW: subscriptions ----

@app.get("/subs")
def get_subs():
    return {"items": service.list_subscriptions()}

@app.post("/subs/subscribe")
def subs_subscribe(req: SubscribeRequest):
    return service.subscribe(req.channel_id, req.channel_title, req.channel_url)

@app.post("/subs/unsubscribe")
def subs_unsubscribe(req: UnsubscribeRequest):
    return service.unsubscribe(req.channel_id)

# ---- NEW: controls ----

@app.get("/categories")
def categories_list():
    return {"items": service.list_categories()}

@app.post("/control/trending")
def control_trending(req: TrendRequest):
    return service.set_trending_only(req.on)

@app.post("/control/category")
def control_category(req: CategoryRequest):
    return service.set_active_category(req.category_id)

@app.post("/control/refresh")
def control_refresh():
    service.clear_pagecache()
    return {"ok": True}
