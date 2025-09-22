from fastapi import APIRouter, Depends
from app.services.forecast_service import ForecastService
from app.deps import get_forecast_service

router = APIRouter(prefix="/forecast", tags=["forecast"])

@router.get("/{ticker}")
async def forecast_ticker(ticker: str, horizon_days: int = 7, svc: ForecastService = Depends(get_forecast_service)):
    res = await svc.forecast(ticker, horizon_days=horizon_days)
    return res.dict()