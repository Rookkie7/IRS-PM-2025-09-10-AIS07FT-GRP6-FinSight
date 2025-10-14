from __future__ import annotations
from fastapi import APIRouter, Depends, Query
from typing import Optional, List
from app.services.rec_service import RecService
from app.services.news_service import NewsService
from app.deps import get_rec_service
router = APIRouter(prefix="/rec", tags=["recommendation"])

@router.post("/user")
async def rec_user(payload: dict, svc: RecService = Depends(get_rec_service)):
   ...

def get_service() -> NewsService:
    from app.main import svc
    return svc

@router.get("/user/news")
def rec_for_user(
    user_id: str,
    limit: int = Query(10, ge=1, le=50),
    refresh: bool = Query(False, description="是否在推荐前拉取一小撮实时新闻并入库（免费配额友好）"),
    symbols: Optional[str] = Query(None, description="可选：逗号分隔的 tickers，refresh=1 时用于实时拉取")
):
    service = get_service()
    syms: Optional[List[str]] = None
    if symbols:
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
    items = service.recommend_for_user(user_id=user_id, limit=limit, refresh=refresh, symbols=syms)
    return {"user_id": user_id, "count": len(items), "items": items}
