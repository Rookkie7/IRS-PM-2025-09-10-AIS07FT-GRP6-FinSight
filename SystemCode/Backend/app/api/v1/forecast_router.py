# app/api/v1/forecast_router.py
from __future__ import annotations
from typing import Optional, List, Literal, Any, Annotated
from fastapi import APIRouter, Depends, Query, HTTPException
from datetime import datetime, timezone

from app.adapters.db.database_client import get_mongo_db
from app.services.forecast_service import ForecastService
from app.deps import get_forecast_service
from app.model.models import ForecastResult

import math
import asyncio
from statistics import pstdev
from datetime import datetime, timezone
from typing import Dict, Any, List
from fastapi import Query, HTTPException, Depends
from app.deps import get_forecast_service

router = APIRouter(prefix="/forecast", tags=["forecast"])

MethodType = Literal["arima", "prophet", "lgbm", "lstm", "persistence", "naive-drift", "ma"]
METHOD_REGEX = r"^(arima|prophet|lgbm|lstm|ma|naive-drift|auto|ensemble\([a-zA-Z,\-\s]+\))$"
def _parse_to_utc_dt(any_dt: Any) -> datetime | None:
    """把各种格式的日期统一成带 tzinfo 的 UTC datetime；失败返回 None。"""
    if isinstance(any_dt, datetime):
        dt = any_dt
    else:
        try:
            # 兼容 '...Z'
            dt = datetime.fromisoformat(str(any_dt).replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt
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
def _directional_accuracy(y: List[float], window: int = 90) -> float:
    if len(y) < 3:
        return 0.7
    window = max(10, min(window, len(y) - 1))
    diffs = [y[i] - y[i-1] for i in range(1, len(y))]
    recent = diffs[-window:]
    if len(recent) < 2:
        return 0.7
    hits = sum((recent[i-1] >= 0) == (recent[i] >= 0) for i in range(1, len(recent)))
    return round(hits / (len(recent) - 1), 4)

def _risk_level(y: List[float]) -> str:
    if len(y) < 20:
        return "Medium"
    rets = [(y[i]/y[i-1] - 1.0) for i in range(1, len(y)) if y[i-1] != 0]
    vol = pstdev(rets) if len(rets) > 1 else 0.0
    if vol < 0.012:
        return "Low"
    if vol < 0.025:
        return "Medium"
    return "High"

def _best_timeframe_from_conf(conf: Dict[int, float] | None, horizons: List[int]) -> str:
    def label(h: int) -> str:
        return "1 Day" if h == 1 else ("1 Week" if h == 7 else f"{h} Days")
    if conf and isinstance(conf, dict) and len(conf):
        best_h = max(conf.items(), key=lambda kv: kv[1])[0]
        return label(int(best_h))
    # 没有置信度就按常见优先级
    if 1 in horizons: return "1 Day"
    if 7 in horizons: return "1 Week"
    return label(sorted(horizons)[0] if horizons else 1)

@router.get(
    "/diagnostics/{ticker}",
    summary="返回模型诊断：accuracy、risk_level、best_timeframe、data_points",
)
async def forecast_diagnostics(
    ticker: str,
    method: str = Query("lstm", description="预测方法"),
    horizons: str = Query("1,2,3,4,5,6,7", description="用于评估 best_timeframe"),
    window: int = Query(90, description="accuracy 回测窗口（天）"),
    svc: ForecastService = Depends(get_forecast_service),
):
    """
    仅返回 4 个卡片指标；不改动现有 /forecast/{ticker} 等接口。
    - 历史数据统一通过 price_provider 获取
    - 预测置信度通过 svc.forecast 解析，最大兼容不同 forecaster 返回结构
    """
    # 1) 历史数据（近 3 年，够算波动与方向准确率）
    closes = await svc.price_provider.get_recent_closes(ticker, lookback_days=365 * 3)
    if not closes or len(closes) < 20:
        raise HTTPException(400, "Not enough history for diagnostics")

    y = [float(x) for x in closes if isinstance(x, (int, float))]

    # 2) 计算 accuracy & risk_level
    accuracy = _directional_accuracy(y, window=window)
    risk = _risk_level(y)

    # 3) 解析 horizons
    H = [int(h) for h in str(horizons).replace(" ", "").split(",") if h.strip().isdigit()]
    if not H:
        H = [1, 2, 3, 4, 5, 6, 7]

    # 4) 通过 svc.forecast 获取置信度（尽力解析）
    conf_by_h: Dict[int, float] | None = None
    try:
        res = await svc.forecast(ticker=ticker, horizon_days=max(H), horizons=H, method=method)
        # res 可能是 Pydantic 模型或 dict
        obj = res.dict() if hasattr(res, "dict") else (res or {})
        # 常见落点1：meta.confidence_by_horizon: {7: 0.72, 30: 0.66, ...}
        conf_by_h = (
            obj.get("meta", {}).get("confidence_by_horizon")
            or obj.get("confidence_by_horizon")
            or None
        )

        # 常见落点2：predictions: [{period:"1 Week", confidence:0.73, ...}, ...]
        if not conf_by_h:
            preds = obj.get("predictions") or obj.get("forecast") or obj.get("points") or []
            if isinstance(preds, list) and preds:
                tmp: Dict[int, float] = {}
                for p in preds:
                    # period 可能是 "1 Day"/"1 Week"/"30 Days" 或数值 horizon
                    period = (p.get("period") or p.get("horizon") or p.get("horizon_days") or "").strip()
                    conf = p.get("confidence")
                    # 尝试从 period 提取整数天
                    h_val: int | None = None
                    s = str(period).lower()
                    if s.isdigit():
                        h_val = int(s)
                    elif "week" in s:
                        h_val = 7
                    elif "day" in s:
                        try:
                            h_val = int("".join(ch for ch in s if ch.isdigit()) or "1")
                        except Exception:
                            h_val = 1
                    elif "month" in s or s == "1m":
                        h_val = 30

                    if h_val is not None and isinstance(conf, (int, float)):
                        tmp[h_val] = float(conf)

                conf_by_h = tmp or None

    except Exception:
        conf_by_h = None  # 失败容错

    best_timeframe = _best_timeframe_from_conf(conf_by_h, H)

    # 5) 返回
    return sanitize_for_json({
        "symbol": ticker.upper(),
        "method": method,
        "accuracy": accuracy,                # 0~1
        "risk_level": risk,                  # "Low"/"Medium"/"High"
        "best_timeframe": best_timeframe,    # "1 Day" / "1 Week" / "N Days"
        "data_points": len(y),               # 历史样本点数
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {"horizons": H, "has_confidence": bool(conf_by_h)},
    })



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
def _to_datestr_yyyy_mm_dd(dt: datetime) -> str:
    return dt.date().isoformat()  # 'YYYY-MM-DD'

@router.get("/prices7", summary="获取指定股票最近7条收盘价（按日期升序）")
async def get_last_7_prices(ticker: Annotated[str, Query(description="如 AAPL")]):
    db = get_mongo_db()
    col = db["stocks"]

    doc = await col.find_one(
        {"symbol": ticker.upper()},
        {"_id": 0, "symbol": 1, "historical_data.time_series": 1}
    )
    if not doc:
        raise HTTPException(status_code=404, detail=f"Stock '{ticker}' not found")

    ts = (doc.get("historical_data") or {}).get("time_series") or []
    if not isinstance(ts, list) or not ts:
        raise HTTPException(status_code=404, detail=f"No time_series for '{ticker}'")

    # === 关键：按“自然日”去重，只保留当天最后一条 close；过滤今天/未来 ===
    today_utc = datetime.now(timezone.utc).date()
    by_day: Dict[str, Dict[str, Any]] = {}

    for idx, rec in enumerate(ts):
        if not isinstance(rec, dict):
            continue
        dt = _parse_to_utc_dt(rec.get("date"))
        if dt is None:
            continue
        try:
            close = float(rec.get("close"))
        except Exception:
            continue

        d_str = _to_datestr_yyyy_mm_dd(dt)  # 'YYYY-MM-DD'
        # 过滤今天与未来（历史段不包含今天）
        if datetime.strptime(d_str, "%Y-%m-%d").date() >= today_utc:
            continue

        # 对同一天多条记录：保留“索引更大/时间更晚”的那条
        exist = by_day.get(d_str)
        if (exist is None) or (idx > exist["_idx"]):
            by_day[d_str] = {"date": d_str, "close": close, "_idx": idx}

    if not by_day:
        raise HTTPException(status_code=404, detail=f"No valid past-day (date,close) bars for '{ticker}'")

    # 取最近 7 个“自然日”（交易日），并按日期升序返回
    days_sorted = sorted(by_day.keys())             # 升序
    last7_keys = days_sorted[-7:]                   # 最近 7 天
    bars = [{"date": k, "close": float(by_day[k]["close"])} for k in last7_keys]

    return {
        "symbol": ticker.upper(),
        "count": len(bars),
        "prices": bars,   # 形如 [{'date':'YYYY-MM-DD','close':123.45}, ...]，日期升序
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


def _to_epoch_s(dt_str: str, i_fallback: int) -> float:
    """
    尝试把各种 ISO 字符串转成 epoch 秒；失败则返回一个基于原始顺序的兜底值，
    确保 key 都是 float，避免类型混排导致排序报错。
    """
    try:
        # 支持 "...Z" 以及无时区形式
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        # fallback: 用一个很小的偏移保证原顺序
        return float(i_fallback)

