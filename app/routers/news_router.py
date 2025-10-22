from __future__ import annotations
import requests
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional, Literal
from app.services.news_service import NewsService
from app.domain.models import NewsItem
from app.utils.news_seed import SEED_NEWS

def get_service() -> NewsService:
    from app.main import svc
    return svc

router = APIRouter(prefix="/api/news", tags=["news"])

@router.post("/ingest/seed")
def ingest_seed(service: NewsService = Depends(get_service)):
    service.ingest([NewsItem(**n) for n in SEED_NEWS])
    return {"ok": True, "count": len(SEED_NEWS)}

@router.get("/feed")
def get_feed(user_id: str = "demo", 
             limit: int = 20,
             strategy: Literal["auto", "personalized", "trending"] = Query("auto"),
             service: NewsService = Depends(get_service)):
    items = service.personalized_feed(user_id=user_id, limit=limit, strategy=strategy)
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

