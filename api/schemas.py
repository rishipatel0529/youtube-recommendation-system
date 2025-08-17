"""
schemas.py
Pydantic request/response models for the YT Recommender API.
Defines typed schemas for recommendations, search, interactions,
subscriptions, favorites, and control endpoints.
"""

from pydantic import BaseModel
from typing import List, Optional

class RecItem(BaseModel):
    # Minimal fields needed by the UI to render a recommendation card.
    video_id: str
    title: str
    channel: str
    channel_id: Optional[str] = None
    duration_sec: int
    why: List[str]
    rank: int
    variant: str

class RecRequest(BaseModel):
    # Recommendation request with pagination and AB variant.
    user_id: str = "local"
    page: int = 1
    page_size: int = 10
    variant: str = "A"

class SearchRequest(BaseModel):
    # Search request with pagination.
    q: str
    page: int = 1
    page_size: int = 10

class InteractRequest(BaseModel):
    # User interaction event (watch/like/skip/etc.).
    user_id: str = "local"
    video_id: str
    event: str
    rank: Optional[int] = None
    variant: Optional[str] = None
    dwell_sec: Optional[float] = None
    page: Optional[int] = None
    is_search: int = 0
    query: Optional[str] = None

class BlockRequest(BaseModel):
    # Block a channel by id (optional title for nicer UX).
    channel_id: str
    channel_title: Optional[str] = None

class SaveVideoRequest(BaseModel):
    # Save a single video to favorites.
    video_id: str

class SaveChannelRequest(BaseModel):
    # Save a channel to favorites.
    channel_id: str
    channel_title: Optional[str] = None

class SubscribeRequest(BaseModel):
    # Create a local 'subscription' entry (not tied to YT account).
    channel_id: str
    channel_title: Optional[str] = None
    channel_url: Optional[str] = None

class UnsubscribeRequest(BaseModel):
    # Remove a local 'subscription' entry.
    channel_id: str

class TrendRequest(BaseModel):
    # Toggle trending-only mode.
    on: bool

class CategoryRequest(BaseModel):
    # Set or clear the active category filter.
    category_id: Optional[str] = None