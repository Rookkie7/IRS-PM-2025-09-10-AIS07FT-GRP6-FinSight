# app/api/v1/forecast_router.py
from __future__ import annotations
from typing import Optional, List, Literal, Any
from fastapi import APIRouter, Depends, Query, HTTPException

from app.adapters.db.database_client import get_mongo_db
from app.services.forecast_service import ForecastService
from app.deps import get_forecast_service
import math
import asyncio

router = APIRouter(prefix="/forecast", tags=["forecast"])

MethodType = Literal["arima", "prophet", "lgbm", "lstm", "persistence", "naive-drift", "ma"]

def sanitize_for_json(obj: Any) -> Any:
    """递归将 NaN/±Inf → None，避免 JSON 序列化报错"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj

@router.get("/get")
async def get_forecast(limit: int = 10):
    db = get_mongo_db()
    col = db['stocks']

    cursor = col.find({}, {"_id": 0}).sort("symbol", 1)
    docs = await cursor.to_list()

    if not docs:
        raise HTTPException(status_code=404, detail="No documents found.")

    docs = sanitize_for_json(docs)

    return {"count": len(docs), "data": docs}

@router.get("/stock")
async def get_stock_info(ticker: str):
    """
    获取指定股票的基本信息
    """
    db = get_mongo_db()
    col = db["stocks"]

    # 转大写防止大小写不一致
    doc = await col.find_one({"symbol": ticker.upper()}, {"_id": 0})

    if not doc:
        raise HTTPException(status_code=404, detail=f"Stock '{ticker}' not found")

    return {"symbol": ticker.upper(), "info": doc}


@router.get("/batch", summary="批量获取股票并做预测")
async def forecast_batch(
    limit: int = Query(100, ge=1, le=1000, description="最多处理多少支股票"),
    symbols: Optional[str] = Query(None, description="逗号分隔: AAPL,TSLA,NVDA；传了就只用这些"),
    horizon_days: int = Query(7, ge=1, le=365),
    horizons: Optional[str] = Query(None, description="逗号分隔: 7,30,90,180"),
    method: MethodType | str = Query("naive-drift", description="arima / naive-drift / ma"),
    concurrency: int = Query(8, ge=1, le=64, description="并发度（根据CPU/IO调整）"),
    svc: ForecastService = Depends(get_forecast_service),
):
    """
    1) 从 MongoDB(stocks) 读取股票代码（或使用 query 指定的 symbols）
    2) 并发调用 ForecastService.forecast 对每支股票进行预测
    3) 汇总返回 { ok: [...], fail: [...] }
    """
    # 解析 horizons
    hs: Optional[List[int]] = None
    if horizons:
        hs = [int(x) for x in horizons.split(",") if x.strip().isdigit()]

    # 准备 symbols 列表
    if symbols:
        tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        # 从 Mongo 读取
        db = get_mongo_db()
        col = db["stocks"]
        # 只取 symbol 字段，避免传输大文档
        cursor = col.find({}, {"_id": 0, "symbol": 1}).sort("symbol", 1).limit(limit)
        docs = await cursor.to_list(length=limit)
        if not docs:
            raise HTTPException(status_code=404, detail="No stocks found in MongoDB.")
        tickers = [d.get("symbol") for d in docs if d.get("symbol")]

    # 并发预测（用信号量控制并发度，避免打爆CPU/IO）
    sem = asyncio.Semaphore(concurrency)
    results_ok: list[dict] = []
    results_fail: list[dict] = []

    async def _one(tk: str):
        async with sem:
            try:
                res = await svc.forecast(
                    tk,
                    horizon_days=horizon_days,
                    horizons=hs,
                    method=method,   # 你的 service 已支持
                )
                # res 可能是 Pydantic 模型 → dict
                out = res.dict() if hasattr(res, "dict") else res
                out = sanitize_for_json(out)
                results_ok.append({"ticker": tk, "result": out})
            except Exception as e:
                results_fail.append({"ticker": tk, "error": str(e)})

    await asyncio.gather(*[_one(t) for t in tickers])

    return {
        "requested": len(tickers),
        "succeeded": len(results_ok),
        "failed": len(results_fail),
        "ok": results_ok,
        "fail": results_fail,
    }

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



