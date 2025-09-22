from fastapi import APIRouter, Depends
from app.services.news_service import NewsService
from app.services import get_news_service

router = APIRouter(prefix="/news", tags=["news"])

@router.post("/ingest")
async def ingest_news(items: list[dict], svc: NewsService = Depends(get_news_service)):
    ...

@router.get("/news")
async def get_news(items: list[dict] = Depends(get_news_service)):
    ...