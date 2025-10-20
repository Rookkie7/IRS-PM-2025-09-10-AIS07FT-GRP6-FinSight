# app/api/v1/forecast_router.py
from __future__ import annotations
from typing import Optional, List, Literal, Any, Annotated
from fastapi import APIRouter, Depends, Query, HTTPException

from app.adapters.db.database_client import get_mongo_db
from app.services.forecast_service import ForecastService
from app.deps import get_forecast_service
from app.model.models import ForecastResult

import math
import asyncio

router = APIRouter(prefix="/forecast", tags=["forecast"])

MethodType = Literal["arima", "prophet", "lgbm", "lstm", "persistence", "naive-drift", "ma"]
METHOD_REGEX = r"^(arima|prophet|lgbm|lstm|ma|naive-drift|auto|ensemble\([a-zA-Z,\-\s]+\))$"

def _parse_horizons(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            v = int(x)
            if v > 0:
                out.append(v)
        except ValueError:
            continue
    return out or None


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
async def get_forecast(limit: int = 100):
    db = get_mongo_db()
    col = db['stocks']

    cursor = col.find({}, {"_id": 0}).sort("symbol", 1).limit(limit)
    docs = await cursor.to_list()

    if not docs:
        raise HTTPException(status_code=404, detail="No documents found.")

    docs = sanitize_for_json(docs)

    return {"count": len(docs), "data": docs}
'''
"symbol": "AAPL",
            "basic_info": {
                "name": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "market_cap": 3700302807040,
                "country": "United States",
                "currency": "USD"
            },
            "created_at": "2025-10-16T08:41:38.465000",
            "descriptions": {
                "business_summary": "..."
            },
            "financials": {
                "key_ratios": {
                    "profit_margin": 0.24295999,
                    "revenue_growth": 0.096,
                    "return_on_equity": 1.49814,
                    "debt_to_equity": 154.486,
                    "current_ratio": 0.868,
                    "operating_margin": 0.29990998,
                    "gross_margin": 0.46678,
                    "earnings_growth": 0.121,
                    "beta": 1.094
                },
                "valuation_metrics": {
                    "market_cap": 3700302807040,
                    "trailing_pe": 37.893616,
                    "forward_pe": 30.004812,
                    "price_to_sales": 9.055498,
                    "price_to_book": 56.271717,
                    "enterprise_value": 3746628894720
                },
                "dividend_info": {
                    "dividend_yield": 0.42,
                    "payout_ratio": 0.1533
                }
            },
            "historical_data": {
                "time_series": [
                    {
                        "date": "2024-10-16T04:00:00",
                        "open": 230.52713995218934,
                        "close": 230.706298828125,
                        "high": 231.04472016362837,
                        "low": 228.77528327315235,
                        "volume": 34082200
                    },
                    {
                        "date": "2024-10-17T04:00:00",
                        "open": 232.348638945834,
                        "close": 231.07456970214844,
                        "high": 232.76670668063704,
                        "low": 229.4521309302649,
                        "volume": 32993800
                    },
                    {...}
                    
                "volatility_30d": 0.24750883337798193,
                "volatility_90d": 0.23385265692524293,
                "momentum_1m": 0.04698720440139992,
                "momentum_3m": 0.193739974393474,
                "volume_avg_30d": 53881043.333333336
            },
            "updated_at": "2025-10-16T08:41:38.465000"
        }
    ]
'''

@router.get("/debug/series")
async def debug_series(ticker: str, svc: ForecastService = Depends(get_forecast_service)):
    closes = await svc.price_provider.get_recent_closes(ticker, lookback_days=252)
    return {"ticker": ticker, "n": len(closes), "head": closes[:3], "tail": closes[-3:]}

@router.get("", summary="对单支股票进行预测（键值对查询）")
async def forecast_query(
    ticker: Annotated[str, Query(description="如 AAPL")],
    horizon_days: Annotated[int, Query(ge=1, le=365)] = 7,
    horizons: Annotated[Optional[str], Query(description="逗号分隔: 7,30,90,180")] = "1,2,3,4,5,7,30,90,180",
    method: Annotated[str, Query(
        pattern=METHOD_REGEX,
        description="arima / prophet / lgbm / lstm / ma / naive-drift / auto / ensemble(a,b,...)"
    )] = "naive-drift",
    svc = Depends(get_forecast_service),
):
    hs = [int(x) for x in horizons.split(",") if x.strip().isdigit()] if horizons else None
    try:
        res = await svc.forecast(ticker=ticker, horizon_days=horizon_days, horizons=hs, method=method)
        # ---- 👇 合并公司名 ----
        try:
            db = get_mongo_db()
            doc = await db["stocks"].find_one(
                {"symbol": ticker.upper()},
                {"_id": 0, "basic_info.name": 1}
            )
            company_name = (doc or {}).get("basic_info", {}).get("name")
            # res 可能是 pydantic 模型或 dict；都处理一下
            if hasattr(res, "company_name"):
                res.company_name = company_name
            elif isinstance(res, dict):
                res["company_name"] = company_name
        except Exception:
            pass
        return sanitize_for_json(res.dict() if hasattr(res, "dict") else res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    horizons: Optional[str] = Query("7,30,90,180", description="逗号分隔: 7,30,90,180"),
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
        # print(tickers)

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

@router.get("/{ticker}", response_model=ForecastResult)
async def forecast_ticker(
    ticker: str,
    horizon_days: int = Query(7, ge=1, le=365, description="若不传 horizons 时的默认单周期"),
    horizons: Optional[str] = Query(
        None,
        description="逗号分隔的多个周期，如 '7,30,90,180'；传了就忽略 horizon_days"
    ),
    method: str = Query(
        "naive-drift",
        description=(
            "预测方法：'naive-drift' | 'ma' | 'arima' | 'prophet' | 'lgbm' | 'lstm' | "
            "'seq2seq' | 'dilated_cnn' | 'transformer' | 'stacked' | 'auto' | "
            "集合形式 'ensemble(a,b,c)'（存在即用，缺失跳过）"
        ),
    ),
    svc: ForecastService = Depends(get_forecast_service),
):
    hs = _parse_horizons(horizons) or [horizon_days]
    return await svc.forecast(ticker, horizon_days=horizon_days, horizons=hs, method=method)


