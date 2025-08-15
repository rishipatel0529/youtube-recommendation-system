from pydantic import BaseModel
from typing import List, Optional

class RecItem(BaseModel):
    video_id: str
    title: str
    channel: str
    channel_id: Optional[str] = None
    duration_sec: int
    why: List[str]
    rank: int
    variant: str

class RecRequest(BaseModel):
    user_id: str = "local"
    page: int = 1
    page_size: int = 10
    variant: str = "A"

class SearchRequest(BaseModel):
    q: str
    page: int = 1
    page_size: int = 10

class InteractRequest(BaseModel):
    user_id: str = "local"
    video_id: str
    event: str
    rank: Optional[int] = None
    variant: Optional[str] = None
    dwell_sec: Optional[float] = None
    page: Optional[int] = None
    is_search: int = 0
    query: Optional[str] = None

# --- Actions / Controls ---

class BlockRequest(BaseModel):
    channel_id: str
    channel_title: Optional[str] = None

class SaveVideoRequest(BaseModel):
    video_id: str

class SaveChannelRequest(BaseModel):
    channel_id: str
    channel_title: Optional[str] = None

class SubscribeRequest(BaseModel):
    channel_id: str
    channel_title: Optional[str] = None
    channel_url: Optional[str] = None

class UnsubscribeRequest(BaseModel):
    channel_id: str

class TrendRequest(BaseModel):
    on: bool

class CategoryRequest(BaseModel):
    category_id: Optional[str] = None
