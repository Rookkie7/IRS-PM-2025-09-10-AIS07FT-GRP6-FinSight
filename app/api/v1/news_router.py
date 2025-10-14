from __future__ import annotations
from fastapi import APIRouter, Depends, Body, Query, HTTPException
import requests
from requests import RequestException
from app.services.news_service import NewsService
# from app.services import get_news_service
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from app.domain.models import NewsItem
from app.utils.news_seed import SEED_NEWS
from app.config import settings

from app.adapters.fetchers.marketaux_fetcher import MarketauxFetcher, MarketauxConfig
from app.adapters.fetchers.rss_fetcher import RSSFetcher
from app.utils.ticker_mapping import load_watchlist_simple
from app.services.ingest_pipeline import IngestPipeline
from requests.exceptions import HTTPError
from fastapi.responses import JSONResponse

def get_service() -> NewsService:
    from app.main import svc
    return svc

router = APIRouter(prefix="/news", tags=["news"])

@router.post("/ingest/seed")
def ingest_seed(service: NewsService = Depends(get_service)):
    service.ingest([NewsItem(**n) for n in SEED_NEWS])
    return {"ok": True, "count": len(SEED_NEWS)}
# @router.post("/ingest")
# async def ingest_news(items: list[dict], svc: NewsService = Depends(get_news_service)):
#     ...

# @router.get("/news")
# async def get_news(items: list[dict] = Depends(get_news_service)):
#     ...

@router.get("/feed")
def get_feed(user_id: str = "demo", limit: int = 20, 
             strategy: Literal["auto", "personalized", "trending"] = Query("auto"),
             service: NewsService = Depends(get_service)):
    items = service.personalized_feed(user_id=user_id, limit=limit)
    # 展示相似度/时间融合分数，便于调试与演示“为何出现”
    out: List[Dict[str, Any]] = []
    for it, score in items:
        out.append({
            "news_id": it.news_id,
            "title": it.title,
            "source": it.source,
            "published_at": it.published_at,
            "tickers": it.tickers,
            "topics": it.topics,
            "sentiment": it.sentiment,
            "score": round(score, 4)
        })
    return {"user_id": user_id, "items": out}

# ---- 新增：通用 ingest schema ----
class IngestNewsItem(BaseModel):
    news_id: str
    title: str
    text: str = ""
    source: str = ""
    published_at: datetime
    tickers: List[str] = Field(default_factory=list)
    topics:  List[str] = Field(default_factory=list)
    sentiment: float = 0.0  # 若无情绪可传0

@router.post("/ingest")
def ingest_news(items: List[IngestNewsItem], service: NewsService = Depends(get_service)):
    # 将外部抓取产物写入（清洗/向量化在 service.ingest 内完成）
    to_save = [NewsItem(**i.model_dump()) for i in items]
    service.ingest(to_save)
    return {"ok": True, "count": len(to_save)}

# 保留种子导入（演示/回归）
@router.post("/ingest/seed")
def ingest_seed(service: NewsService = Depends(get_service)):
    service.ingest([NewsItem(**n) for n in SEED_NEWS])
    return {"ok": True, "count": len(SEED_NEWS)}

@router.post("/ingest/pull")
def ingest_pull(
    source: str = Query(..., description="marketaux | rss"),
    region: Optional[str] = Query(None, description="us | in | None"),
    symbols: Optional[str] = Query(None, description="逗号分隔，如 AAPL,NVDA,TCS.NS"),
    since_hours: int = Query(6, ge=1, le=48),
    preview: bool = Query(False, description="true 则只返回采样预览，不入库"),
    service: NewsService = Depends(get_service),
):
    """
    手动触发抓取并入库：
    - source=marketaux: 支持 symbols & since_hours
    - source=rss: 拉环境配置中的 RSS 列表
    - preview=true: 仅返回前20条预览，不入库
    """
    # 读取 watchlist（有错则忽略）
    try:
        watch = load_watchlist_simple(settings.WATCHLIST_FILE) if getattr(settings, "WATCHLIST_FILE", None) else {}
    except Exception:
        watch = {}

    pipe = IngestPipeline(news_repo=service.news_repo, embedder=service.embedder, watchlist=watch)

    try:
        if source.lower() == "marketaux":
            if not settings.MARKETAUX_API_KEY:
                raise HTTPException(400, "MARKETAUX_API_KEY not set")

            # 解析 symbols（query > env > region 兜底）
            syms = None
            if symbols:
                syms = [s.strip() for s in symbols.split(",") if s.strip()]
            if not syms:
                default_env = (getattr(settings, "MARKETAUX_DEFAULT_SYMBOLS", "") or "").strip()
                if default_env:
                    syms = [s.strip() for s in default_env.split(",") if s.strip()]
            if not syms:
                syms = ["TCS.NS", "INFY.NS", "RELIANCE.NS"] if (region and region.lower() == "in") else ["AAPL", "NVDA", "MSFT"]

            mcfg = MarketauxConfig(
                api_key=settings.MARKETAUX_API_KEY,
                qps=float(getattr(settings, "FETCH_QPS", 0.5)),
                daily_budget=int(getattr(settings, "DAILY_BUDGET_MARKETAUX", 80)),
                page_size=3,
            )
            fetcher = MarketauxFetcher(mcfg)

            try:
                raw = fetcher.pull_recent(
                    symbols=syms,
                    since_hours=since_hours,
                    region=region,
                    max_pages=2,
                )
            except requests.HTTPError as ex:  # 只拦上游 HTTP 错误，直透信息
                status = getattr(ex.response, "status_code", 0)
                body = getattr(ex.response, "text", "") or ""
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=502,
                    content={
                        "ok": False,
                        "source": "marketaux",
                        "upstream_status": status,
                        "upstream_body_preview": body[:800],
                        "hint": "检查 API key / 配额 / 参数。免费层通常限制返回条数与速率。"
                    },
                )
        
        # elif source.lower() == "rss":
        #     feeds: List[str] = []
        #     if region in (None, "us"):
        #         feeds += [x.strip() for x in (getattr(settings, "RSS_SOURCES_US", "") or "").split(",") if x.strip()]
        #     if region in (None, "in"):
        #         feeds += [x.strip() for x in (getattr(settings, "RSS_SOURCES_IN", "") or "").split(",") if x.strip()]
        #     # 去重保持顺序
        #     feeds = list(dict.fromkeys(feeds))
        #     fetcher = RSSFetcher(qps=float(getattr(settings, "RSS_QPS", 1.0)))
        #     raw = fetcher.pull_many(feeds, limit_per_feed=30)

        else:
            raise HTTPException(400, "source must be marketaux | rss")

    except HTTPError as e:
        status = getattr(e.response, "status_code", 0)
        body   = getattr(e.response, "text", "") or str(e)
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "source": source,
                "upstream_status": status,
                "upstream_body_preview": body[:1000],
                "hint": "检查 API key / 配额 / 参数。免费层通常限制返回条数与速率。",
            },
        )
    except RequestException as ex:
        return JSONResponse(
            status_code=502,
            content={"ok": False, "source": source, "error": "request_exception", "message": str(ex)},
        )
    except Exception as ex:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "internal_ingest_error", "message": str(ex)},
        )

    # 正常返回
    if preview:
        return {"ok": True, "count": len(raw), "sample": raw[:20]}

    dedup, stored = pipe.ingest_dicts(raw)
    return {"ok": True, "pulled": len(raw), "ingested": stored, "dedup_after": dedup}

@router.post("/ingest/preview")
def ingest_preview(items: List[Dict[str, Any]] = Body(...), service: NewsService = Depends(get_service)):
    """
    预览：传入原始字典列表，返回规范化后的样例（不入库、带 32d 向量）
    """
    # watchlist
    watch = {}
    try:
        if getattr(settings, "WATCHLIST_FILE", None):
            from app.utils.ticker_mapping import load_watchlist
            watch = load_watchlist(settings.WATCHLIST_FILE)
    except Exception:
        watch = {}

    from app.services.ingest_pipeline import IngestPipeline
    pipe = IngestPipeline(news_repo=service.news_repo, embedder=service.embedder, watchlist=watch)
    # 仅做规范化与向量，不入库
    from copy import deepcopy
    sample = []
    for it in items[:10]:
        dit = deepcopy(it)
        # 复用内部规范化逻辑
        dit.setdefault("external_id", dit.get("url", ""))
        ni = pipe._to_news_item(dit)
        sample.append(ni.model_dump())
    return {"ok": True, "count": len(sample), "items": sample}