from __future__ import annotations
from typing import List, Any, Optional
import pandas as pd

from app.adapters.db.database_client import get_mongo_db
from app.services.forecast_service import PriceProviderPort

def _dig(d: dict, path: list[str]) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur

def _first_non_empty(lst: list[Any]) -> Any:
    for x in lst:
        if x:
            return x
    return None

class MongoStockPriceProvider(PriceProviderPort):
    """
    更稳健的提取逻辑：
    1) 依次尝试以下路径拿 time_series（list[dict] / dict[date]->dict 均可）：
        - info.historical_data.time_series
        - historical_data.time_series
        - info.time_series
        - time_series
        - prices.time_series
    2) 兼容列名：close / Close / adj_close / adjClose / Adj Close / price
    3) 日期失败则不按日期过滤，直接取尾部 N 条
    """

    def __init__(self, collection_name: str = "stocks"):
        self.collection_name = collection_name

    async def get_recent_closes(self, ticker: str, lookback_days: int = 252) -> List[float]:
        db = get_mongo_db()
        col = db[self.collection_name]

        # 一次性把可能用到的路径都投影出来，避免大文档搬运
        doc = await col.find_one(
            {"symbol": ticker.upper()},
            {
                "_id": 0,
                "symbol": 1,
                "info.symbol": 1,
                "info.historical_data.time_series": 1,
                "historical_data.time_series": 1,
                "info.time_series": 1,
                "time_series": 1,
                "prices.time_series": 1,
            }
        )
        if not doc:
            return []

        # 1) 找到 time_series 容器（list[dict] 或 dict[str]->dict）
        ts: Optional[Any] = _first_non_empty([
            _dig(doc, ["info", "historical_data", "time_series"]),
            _dig(doc, ["historical_data", "time_series"]),
            _dig(doc, ["info", "time_series"]),
            _dig(doc, ["time_series"]),
            _dig(doc, ["prices", "time_series"]),
        ])
        if not ts:
            # 打印一下方便调试（uvicorn 控制台可见）
            print(f"[PriceProvider] {ticker}: no time_series found in known paths.")
            return []

        # 2) 统一转为 list[dict]
        if isinstance(ts, dict):
            # 形如 { "2024-10-16": {"close": ...}, ... }
            ts_list = []
            for k, v in ts.items():
                if isinstance(v, dict):
                    rec = {"date": k, **v}
                    ts_list.append(rec)
            ts = ts_list

        if not isinstance(ts, list) or not ts:
            print(f"[PriceProvider] {ticker}: time_series not list or empty.")
            return []

        df = pd.DataFrame(ts)

        # 3) 识别列名（close 的各种别名）
        cand_close = [c for c in ["close", "Close", "adj_close", "adjClose", "Adj Close", "price"] if c in df.columns]
        if not cand_close:
            print(f"[PriceProvider] {ticker}: no close-like column in {list(df.columns)}")
            return []
        close_col = cand_close[0]

        # 4) 日期解析 + 排序；失败则走尾部窗口回退
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
            df = df.dropna(subset=["date"]).sort_values("date")
            if not df.empty:
                # 正常按自然日窗口过滤
                try:
                    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
                    df = df[df["date"] >= cutoff]
                except Exception:
                    pass
        else:
            # 没有 date 列，保持原顺序（假定已按时间升序）
            pass

        # 如果经过日期过滤后空了，做尾部窗口回退（最多 300 条防止超大）
        if df.empty:
            df = pd.DataFrame(ts)
            if close_col not in df.columns:
                return []
            df = df.tail(min(lookback_days, 300))

        closes = pd.to_numeric(df[close_col], errors="coerce").dropna().tolist()

        # 调试输出（只打印前后各1个）
        if not closes:
            print(f"[PriceProvider] {ticker}: closes empty after normalization. cols={list(df.columns)}")
        else:
            print(f"[PriceProvider] {ticker}: closes len={len(closes)} sample=({closes[0]}, {closes[-1]})")

        return closes
