from __future__ import annotations
from fastapi import APIRouter, Depends, Query
from typing import Optional, List
from app.services.rec_service import RecService
from app.services.news_service import NewsService
from app.deps import get_rec_service

from app.config import settings

import logging
log = logging.getLogger("app.rec.debug")

EXCLUDE_HOURS = int(getattr(settings, "RECENT_EXCLUDE_HOURS", 72))
router = APIRouter(prefix="/rec", tags=["recommendation"])

@router.post("/user")
async def rec_user(payload: dict, svc: RecService = Depends(get_rec_service)):
   ...

def get_service() -> NewsService:
    from app.main import svc
    return svc

@router.get("/user/news")
# def user_news(
#     user_id: str,
#     limit: int = 20,
#     refresh: int = 0,
#     exclude_hours: int | None = 720,  # 30天内的“已浏览”都过滤；你也可传更小
#     symbols: str | None = None,       # 可选：透传 symbols，逗号分隔
#     svc: NewsService = Depends(get_service),
# ):
#     syms = [s.strip().upper() for s in symbols.split(",")] if symbols else None
#     items = svc.recommend_for_user(
#         user_id=user_id,
#         limit=int(limit),
#         refresh=bool(int(refresh)),
#         exclude_hours=exclude_hours,
#         symbols=syms,                  # 关键：把 symbols 传进去（让 refresh 真按你想要的拉）
#     )
#     return {"user_id": user_id, "count": len(items), "items": items}
# 以上是可行版本
@router.get("/rec/user/news")
def rec_user_news(
    user_id: str,
    limit: int = 20,
    refresh: int = 0,              # 允许 0/1
    symbols: Optional[str] = None, # 逗号分隔
    exclude_hours: Optional[int] = None,
    svc: NewsService = Depends(get_service),
):
    sym_list = [s.strip().upper() for s in symbols.split(",")] if symbols else None

    items = svc.recommend_for_user(
        user_id=user_id,
        limit=limit,
        refresh=bool(int(refresh)),   # ✅ 确保转成 True/False
        symbols=sym_list,             # ✅ 传给 service（再传给 fetcher）
        exclude_hours=exclude_hours,  # ✅ 继续传到过滤层
    )
    return {"user_id": user_id, "count": len(items), "items": items}

@router.get("/debug/ping_fetch")
def debug_ping_fetch(symbols: Optional[str] = None):
    # 只测试拉取，不入库
    from app.adapters.fetchers.marketaux_fetcher import MarketauxFetcher, MarketauxConfig
    from app.config import settings
    mcfg = MarketauxConfig(
        api_key=settings.MARKETAUX_API_KEY,
        qps=float(getattr(settings, "FETCH_QPS", 0.5)),
        daily_budget=int(getattr(settings, "DAILY_BUDGET_MARKETAUX", 80)),
        page_size=3,
    )
    mfetch = MarketauxFetcher(mcfg)
    syms = [s.strip() for s in (symbols or "").split(",") if s.strip()] or None
    items = mfetch.pull_recent(symbols=syms, since_hours=6, max_pages=1)
    return {"fetched": len(items), "sample_titles": [it.get("title") for it in items[:3]]}

# app/api/v1/rec_router.py 内的 debug endpoint
@router.get("/debug/ping_latest")
def debug_ping_latest():
    svc = get_service()
    try:
        # 直接读原始文档，避免 NewsItem/dict 混用
        docs = getattr(svc.news_repo, "raw_latest_docs", lambda n: [])(limit=20)
        # 顺便给个总数
        n = getattr(svc.news_repo, "count_all", lambda: None)()
        return {
            "count_all": n,
            "latest": [{"news_id": d.get("news_id"), "title": d.get("title")} for d in docs]
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
    
@router.get("/debug/where_am_i")
def debug_where():
    from app.main import svc
    return {
        "repo_class": type(svc.news_repo).__name__,
        "has_upsert_many_dicts": hasattr(svc.news_repo, "upsert_many_dicts"),
        "count_all": getattr(svc.news_repo, "count_all", lambda: None)(),
    }

@router.get("/debug/ping_ingest_once")
def debug_ingest_once(symbols: str = "TSLA"):
    from app.main import svc
    from app.adapters.fetchers.marketaux_fetcher import MarketauxFetcher, MarketauxConfig
    from app.config import settings
    from app.services.ingest_pipeline import IngestPipeline

    mcfg = MarketauxConfig(
        api_key=settings.MARKETAUX_API_KEY,
        qps=float(getattr(settings, "FETCH_QPS", 0.5)),
        daily_budget=int(getattr(settings, "DAILY_BUDGET_MARKETAUX", 80)),
        page_size=10,
    )
    fetcher = MarketauxFetcher(mcfg)
    raw = fetcher.pull_recent(symbols=[s.strip() for s in symbols.split(",") if s.strip()], since_hours=24, max_pages=1)
    log.warning(f"[debug_ingest] fetched={len(raw or [])}, sample_title={(raw[0].get('title') if raw else None)}")

    pipe = IngestPipeline(news_repo=svc.news_repo, embedder=svc.embedder, watchlist=None)
    res = pipe.ingest_dicts(raw)

    # 出库校验
    n = getattr(svc.news_repo, "count_all", lambda: None)()
    latest = getattr(svc.news_repo, "raw_latest_docs", lambda n: [])(limit=5)
    return {
        "upsert_result": res,
        "after_count_all": n,
        "latest_titles": [x.get("title") for x in latest],
        "latest_has_prof20": [("vector_prof_20d" in x and len(x.get("vector_prof_20d") or []) == 20) for x in latest],
    }