from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from app.domain.models import UserProfile, BehaviorEvent
from app.services.news_service import NewsService
from pydantic import BaseModel, Field

def get_service() -> NewsService:
    from app.main import svc
    return svc

router = APIRouter(prefix="/api/user", tags=["user"])

@router.post("/profile/init")
def init_profile(user_id: str = "demo", service: NewsService = Depends(get_service)) -> UserProfile:
    prof = service.prof_repo.get_or_create(user_id)
    service.prof_repo.save(prof)
    return prof

@router.post("/event")
def add_event(ev: BehaviorEvent, service: NewsService = Depends(get_service)):
    prof = service.record_event_and_update_profile(ev)
    return {"ok": True, "user_id": ev.user_id, "profile_normed_head": prof.vector[:5]}

class ClickEvent(BaseModel):
    user_id: str = "demo"
    news_id: str
    dwell_ms: int = 0
    liked: bool = False
    bookmarked: bool = False

@router.post("/event/click")
def user_click(ev: ClickEvent, service: NewsService = Depends(get_service)):
    # 取新闻向量（语义 + 画像）
    doc = service.news_repo.get(ev.news_id)
    if not doc:
        raise HTTPException(404, "news not found")

    news_sem = list(getattr(doc, "vector", []) or [])
    news_prof = list(getattr(doc, "vector_prof_20d", []) or [])

    # 简单权重：点击=1，>10s=1.5，like/bookmark 各+0.5
    w = 1.0
    if ev.dwell_ms >= 10000: w += 0.5
    if ev.liked: w += 0.5
    if ev.bookmarked: w += 0.5

    # 更新用户画像
    service.prof_repo.update_user_vectors_from_event(
        user_id=ev.user_id,
        news_sem=news_sem,
        news_prof=news_prof,
        weight=w,
    )
    return {"ok": True, "weight": w}
