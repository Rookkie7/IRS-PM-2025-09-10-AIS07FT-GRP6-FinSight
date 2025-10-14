from __future__ import annotations
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field
from app.domain.models import UserProfile, BehaviorEvent
from app.services.news_service import NewsService
from app.model.models import UserPublic
from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.deps import get_auth_service, get_user_service

def get_service() -> NewsService:
    from app.main import svc
    return svc

router = APIRouter(prefix="/api/user", tags=["user"])

@router.post("/profile/init")
def init_profile(user_id: str = "demo", 
                 reset: bool = Query(False, description="是否重置已有画像为零向量"),
                 service: NewsService = Depends(get_service)) -> UserProfile:
    prof = service.prof_repo.get_or_create(user_id)
    if reset:
        dim = getattr(service.embedder, "dim", len(prof.vector) or 32)
        prof.vector = [0.0]*dim
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

class LikeEvent(BaseModel):
    user_id: str = "demo"
    news_id: str

class BookmarkEvent(BaseModel):
    user_id: str = "demo"
    news_id: str

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

@router.post("/event/like")
def user_like(ev: LikeEvent, service: NewsService = Depends(get_service)):
    doc = service.news_repo.get(ev.news_id)
    if not doc:
        raise HTTPException(404, "news not found")
    news_sem = list(getattr(doc, "vector", []) or [])
    news_prof = list(getattr(doc, "vector_prof_20d", []) or [])

    # like 权重偏低，聚合“认同”信号
    w = 0.7
    service.prof_repo.update_user_vectors_from_event(
        user_id=ev.user_id, news_sem=news_sem, news_prof=news_prof, weight=w,
    )
    return {"ok": True, "weight": w}

@router.post("/event/bookmark")
def user_bookmark(ev: BookmarkEvent, service: NewsService = Depends(get_service)):
    doc = service.news_repo.get(ev.news_id)
    if not doc:
        raise HTTPException(404, "news not found")
    news_sem = list(getattr(doc, "vector", []) or [])
    news_prof = list(getattr(doc, "vector_prof_20d", []) or [])

    # bookmark 权重略高，表示“强偏好/留存”
    w = 1.2
    service.prof_repo.update_user_vectors_from_event(
        user_id=ev.user_id, news_sem=news_sem, news_prof=news_prof, weight=w,
    )
    return {"ok": True, "weight": w}




router = APIRouter(prefix="/users", tags=["users"])

class ProfileUpdateIn(BaseModel):
    full_name: str | None = None
    bio: str | None = None
    interests: list[str] = []
    sectors: list[str] = []
    tickers: list[str] = []

@router.get("/me", response_model=UserPublic)
async def me(auth: AuthService = Depends(get_auth_service)):
    u = await auth.get_current_user()
    return UserPublic(
        id=u.id, email=u.email, username=u.username,
        created_at=u.created_at, profile=u.profile, embedding=u.embedding
    )

@router.put("/me")
async def update_me(payload: ProfileUpdateIn, auth: AuthService = Depends(get_auth_service), usvc: UserService = Depends(get_user_service)):
    u = await auth.get_current_user()
    profile = {
        "full_name": payload.full_name,
        "bio": payload.bio,
        "interests": payload.interests,
        "sectors": payload.sectors,
        "tickers": payload.tickers,
    }
    await usvc.update_profile_and_embed(u, profile)
    return {"ok": True}
