# app/services/macro_service.py
from __future__ import annotations
import os, time, asyncio
from typing import Literal, Dict, Any, Tuple, Optional
import yfinance as yf
import httpx

Provider = Literal["yahoo", "alpha"]

class TTLCache:
    def __init__(self, ttl_sec: int = 30):
        self.ttl = ttl_sec
        self.data: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        if key in self.data:
            ts, val = self.data[key]
            if now - ts < self.ttl:
                return val
        return None

    def set(self, key: str, val: Any):
        self.data[key] = (time.time(), val)

class MacroService:
    """
    提供 ^GSPC（标普）、^VIX（恐慌指数）即时值 + 简易情绪判断。
    默认 provider = yahoo（免 key），可切换 alpha（需 ALPHA_VANTAGE_KEY）。
    """
    def __init__(self, provider: Provider = "yahoo", ttl_sec: int = 30):
        self.provider = provider
        self.alpha_key = os.getenv("ALPHA_VANTAGE_KEY", "")
        self.cache = TTLCache(ttl_sec=ttl_sec)

    async def get_spx(self) -> Dict[str, Any]:
        cache_key = f"spx:{self.provider}"
        hit = self.cache.get(cache_key)
        if hit: return hit

        if self.provider == "alpha":
            data = await self._alpha_quote("^GSPC")  # AlphaVantage 不支持指数符号，见注释
        else:
            data = await self._yahoo_quote("^GSPC")

        self.cache.set(cache_key, data)
        return data

    async def get_vix(self) -> Dict[str, Any]:
        cache_key = f"vix:{self.provider}"
        hit = self.cache.get(cache_key)
        if hit: return hit

        if self.provider == "alpha":
            data = await self._alpha_quote("^VIX")  # 同上，指数符号在 Alpha 需替代，见注释
        else:
            data = await self._yahoo_quote("^VIX")

        self.cache.set(cache_key, data)
        return data

    async def get_sentiment(self) -> Dict[str, Any]:
        """
        非学术版：用 SPX 当日涨跌 & VIX 水平做一个直观标签。
        你也可以替换成你 Mongo 里的新闻情绪聚合。
        """
        spx = await self.get_spx()
        vix = await self.get_vix()
        chg = spx.get("change_pct", 0.0)
        v = vix.get("price", 0.0)

        # 简易规则，可按需微调
        if chg >= 0.3 and v < 15:
            label = "Bullish"
        elif chg <= -0.3 and v > 20:
            label = "Bearish"
        else:
            label = "Neutral"

        return {
            "label": label,
            "inputs": {
                "spx_change_pct": chg,
                "vix": v
            }
        }

    # ---------- Providers ----------
    async def _yahoo_quote(self, symbol: str) -> Dict[str, Any]:
        """
        使用 yfinance 的 1d 历史数据取最新价与前收，避免 .info 的不稳定。
        """
        def _fetch():
            tk = yf.Ticker(symbol)
            # 取最近两天（保证拿到前收），日线
            hist = tk.history(period="5d", interval="1d", auto_adjust=False)
            if hist.empty:
                return {"symbol": symbol, "price": None, "change_pct": None, "ts": None}
            last = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) >= 2 else None
            price = float(getattr(last, "Close", last.get("Close", None)))
            prev_close = float(getattr(prev, "Close", prev["Close"])) if prev is not None else price
            change_pct = (price - prev_close) / prev_close * 100 if prev_close else None
            ts = last.name.isoformat() if hasattr(last, "name") else None
            return {"symbol": symbol, "price": round(price, 4), "change_pct": round(change_pct, 3) if change_pct is not None else None, "ts": ts}
        return await asyncio.to_thread(_fetch)

    async def _alpha_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Alpha Vantage 注意事项：
        - FREE 版对指数支持有限，^GSPC/^VIX 一般无法直接取；可改用 SPY、VIXY 等 ETF 近似。
        - 如果你确实要 Alpha，请把 symbol 换成 'SPY' 和 'VIXY'。
        """
        if not self.alpha_key:
            return {"symbol": symbol, "error": "ALPHA_VANTAGE_KEY missing", "price": None, "change_pct": None, "ts": None}

        # 用 ETF 近似：
        sym_map = {"^GSPC": "SPY", "^VIX": "VIXY"}
        qsym = sym_map.get(symbol, symbol)

        url = "https://www.alphavantage.co/query"
        params = {"function": "GLOBAL_QUOTE", "symbol": qsym, "apikey": self.alpha_key}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            j = r.json()
        q = j.get("Global Quote", {})
        price = float(q.get("05. price")) if q.get("05. price") else None
        change_pct = float(q.get("10. change percent", "0%").strip("%")) if q.get("10. change percent") else None
        ts = q.get("07. latest trading day")

        return {"symbol": symbol, "proxy": qsym, "price": price, "change_pct": change_pct, "ts": ts}