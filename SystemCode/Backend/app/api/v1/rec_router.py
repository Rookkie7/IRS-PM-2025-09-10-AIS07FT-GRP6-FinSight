from __future__ import annotations
from fastapi import APIRouter, Depends, Query
from typing import Optional, List
from app.services.rec_service import RecService
from app.services.news_service import NewsService
from app.deps import get_rec_service

from app.config import settings
EXCLUDE_HOURS = int(getattr(settings, "RECENT_EXCLUDE_HOURS", 72))
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

    # 2) 取到“已排序”的候选（你的 recommend_for_user 已经打分+排序）
    ranked = service.recommend_for_user(
        user_id=user_id,
        limit=limit * 3,      # 拉宽一点，方便过滤后仍有足量
        refresh=refresh,
        symbols=syms
    ) or []

    # 3) 取用户最近交互过的 news_id（默认 72h，可用 .env 的 RECENT_EXCLUDE_HOURS 配置）
    try:
        seen_ids = service.ev_repo.recent_interacted_news_ids(user_id=user_id, since_hours=EXCLUDE_HOURS)
    except Exception:
        # 如果事件仓库不支持该方法或发生异常，则不做过滤
        seen_ids = set()

    def _nid(x):
        # 兼容 dict / pydantic 模型 / 任意带属性对象
        return (isinstance(x, dict) and x.get("news_id")) or getattr(x, "news_id", None)

    # 4) 先取“未交互”的前 N
    primary = []
    for it in ranked:
        nid = _nid(it)
        if not nid:
            continue
        if nid in seen_ids:
            continue
        primary.append(it)
        if len(primary) >= limit:
            break

    # 5) 若数量仍不足，用“已交互但分数高”的补齐（可选）
    items = primary
    if len(items) < limit:
        need = limit - len(items)
        fallback = []
        for it in ranked:
            nid = _nid(it)
            if not nid or nid in {_nid(x) for x in items}:
                continue
            if nid in seen_ids:
                fallback.append(it)
            if len(fallback) >= need:
                break
        items = items + fallback

    # 6) 截断到期望数量并返回
    items = items[:limit]
    return {"user_id": user_id, "count": len(items), "items": items}
