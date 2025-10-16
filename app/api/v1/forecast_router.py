# from fastapi import APIRouter, Depends
# from app.services.forecast_service import ForecastService
# from app.deps import get_forecast_service

# router = APIRouter(prefix="/forecast", tags=["forecast"])

# @router.get("/{ticker}")
# async def forecast_ticker(ticker: str, 
#                           horizon_days: int = 7, 
#                           svc: ForecastService = Depends(get_forecast_service)):
#     res = await svc.forecast(ticker, horizon_days=horizon_days)
#     return res.dict()


# app/api/v1/forecast_router.py
from __future__ import annotations
from typing import Optional, List, Literal
from fastapi import APIRouter, Depends, Query

from app.services.forecast_service import ForecastService
from app.deps import get_forecast_service

router = APIRouter(prefix="/forecast", tags=["forecast"])

# 你也可以把 Literal 去掉，改成 str 自由输入（我这里给了白名单，方便文档与纠错）
MethodType = Literal["arima", "prophet", "lgbm", "lstm", "persistence", "naive-drift", "ma"]

@router.get("/{ticker}")
async def forecast_ticker(
    ticker: str,
    horizon_days: int = 7,
    horizons: Optional[str] = Query(None, description="逗号分隔，如 7,30,90,180"),
    method: MethodType = Query(  # ← 新增：预测方法
        "naive-drift",
        description="预测方法：arima / naive-drift / ma"
    ),
    svc: ForecastService = Depends(get_forecast_service),
):
    hs: Optional[List[int]] = None
    if horizons:
        hs = [int(x) for x in horizons.split(",") if x.strip().isdigit()]

    res = await svc.forecast(
        ticker,
        horizon_days=horizon_days,
        horizons=hs,
        method=method,   # ← 传递给服务层
    )
    return res.dict()
