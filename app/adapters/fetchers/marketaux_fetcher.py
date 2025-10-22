# app/adapters/fetchers/marketaux_fetcher.py
from __future__ import annotations
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union

import requests
from app.config import settings
import logging
log = logging.getLogger("app.adapters.fetchers.marketaux_fetcher")

# 统一 11 行业标签（顺序固定，和 user_profile_20d 前11维一一对应）
INDUSTRIES_11 = [
    "Utilities","Technology","Consumer Defensive","Healthcare","Basic Materials","Real Estate",
    "Energy","Industrials","Consumer Cyclical","Communication Services","Financial Services"
]

# 每个行业的关键词（可逐步扩充）
IND_KEYWORDS = {
    "Utilities": ["utility","utilities","power grid","electric utility","water utility","gas utility"],
    "Technology": ["technology","tech","semiconductor","chip","ai","software","hardware","cloud","saas","gpu","data center"],
    "Consumer Defensive": ["consumer defensive","staples","grocery","beverage","household products","tobacco"],
    "Healthcare": ["healthcare","pharma","biotech","medical device","hospital","drug"],
    "Basic Materials": ["basic materials","mining","chemical","steel","cement","paper","wood"],
    "Real Estate": ["real estate","reit","property","developer"],
    "Energy": ["energy","oil","gas","refinery","renewable","solar","wind"],
    "Industrials": ["industrial","manufacturing","machinery","aerospace","defense","logistics"],
    "Consumer Cyclical": ["consumer cyclical","automotive","retail","discretionary","travel","leisure","luxury","ev"],
    "Communication Services": ["communication services","media","telecom","advertising","streaming","social"],
    "Financial Services": ["financial services","bank","brokerage","insurance","asset management","fintech"]
}

def _normalize_industry_name(s: str) -> str | None:
    """从 Marketaux 的 industry/type/关键词里，映射到 11 行业之一。"""
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip().lower()

    # 1) 先尝试 exact / 同义匹配（常见别名）
    ALIASES = {
        "financials": "Financial Services",
        "financial": "Financial Services",
        "communication": "Communication Services",
        "communications": "Communication Services",
        "comm services": "Communication Services",
        "consumer discretionary": "Consumer Cyclical",
        "staples": "Consumer Defensive",
        "materials": "Basic Materials",
        "tech": "Technology",
        "energy sector": "Energy",
        "industry": "Industrials",
        "health care": "Healthcare",
        "realty": "Real Estate",
        "utilities sector": "Utilities",
    }
    if t in ALIASES:
        return ALIASES[t]

    # 2) 关键词粗归类
    for name, kws in IND_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return name

    # 3) 若本身就是 11 行业之一（忽略大小写）
    for name in INDUSTRIES_11:
        if t == name.lower():
            return name

    return None

@dataclass
class MarketauxConfig:
    api_key: str
    qps: float = 0.5
    daily_budget: int = 80
    page_size: int = 3  # 免费层强制到 <=3；若外部传更大也会被 clamp


class MarketauxFetcher:
    BASE = "https://api.marketaux.com/v1/news/all"

    def __init__(self, cfg_or_key: Union[MarketauxConfig, str], qps: float = 0.5, page_size: int = 20):
        """
        兼容两种用法：
        - MarketauxFetcher(MarketauxConfig(...))
        - MarketauxFetcher(api_key: str, qps=?, page_size=?)
        """
        if isinstance(cfg_or_key, MarketauxConfig):
            cfg = cfg_or_key
            self.api_key = (cfg.api_key or "").strip()
            self.qps = max(float(cfg.qps), 0.05)
            self.page_size = max(1, min(int(cfg.page_size), 3))
        else:
            self.api_key = (cfg_or_key or "").strip()
            self.qps = max(float(qps), 0.05)
            # 免费层友好：limit 最多 3
            self.page_size = max(1, min(int(page_size), 3))

        self._last_ts = 0.0

    def _throttle(self):
        gap = 1.0 / self.qps
        dt = time.time() - self._last_ts
        if dt < gap:
            time.sleep(gap - dt)
        self._last_ts = time.time()

    def _request(self, params: Dict) -> Dict:
        self._throttle()
        params = dict(params)
        params["api_token"] = self.api_key
        resp = requests.get(self.BASE, params=params, timeout=15)
        if not resp.ok:
            # 抛出 HTTPError，由路由层包装为 502 并原样透出上游 body
            raise requests.HTTPError(f"{resp.status_code} {resp.reason}", response=resp)
        return resp.json()

    # === 新增：极轻量关键词 → 主题标签 ===
    def _kw_topics(self, title: str, text: str) -> List[str]:
        """
        返回最多1个行业标签(从 11 行业里选）；找不到则空列表。
        """
        blob = f"{title} {text}".lower()
        hits: list[str] = []
        # 按关键词匹配
        for name, kws in IND_KEYWORDS.items():
            for kw in kws:
                if kw in blob:
                    hits.append(name)
                    break
        # 去重
        hits = list(dict.fromkeys(hits))
        # 只保留一个（可选：按优先级或首个）
        return hits[:1]
    
    def _norm_one(self, it: Dict) -> Dict:
        title = it.get("title") or ""
        # 市面免费层通常不给全文：先 description → snippet
        text = (it.get("description") or it.get("snippet") or "") or ""
        src = (it.get("source") or {}).get("name") if isinstance(it.get("source"), dict) else it.get("source")
        published = it.get("published_at") or it.get("published_at_utc") or it.get("published_at_local")

        entities = it.get("entities") or []
        ent_sents = []
        industries = []

        tickers = []
        for e in entities:
            if not isinstance(e, dict):
                continue
            # 股票代码
            if (e.get("type") or "").lower() == "equity" and e.get("symbol"):
                tickers.append(e["symbol"])
            # 行业与情感
            ind = e.get("industry")
            if isinstance(ind, str) and ind.strip():
                industries.append(ind.strip())
            sc = e.get("sentiment_score")
            if isinstance(sc, (int, float)):
                ent_sents.append(float(sc))

        # topics = 单一 industry；多时取出现最多者
        topic = None
        if industries:
            from collections import Counter
            topic = Counter(industries).most_common(1)[0][0]

        overall_sent = sum(ent_sents) / len(ent_sents) if ent_sents else 0.0

        # 如果 entities 没有 symbol，再退回响应里的 symbols 字段
        symbols = it.get("symbols") or []
        if symbols and isinstance(symbols, list):
            symbols = [s for s in symbols if isinstance(s, str) and s.strip()]
        if not tickers:
            tickers = symbols

        return {
            "news_id": it.get("uuid") or it.get("id") or it.get("url"),
            "title": title,
            "text": text,
            "source": src,
            "url": it.get("url") or "",
            "published_at": published,
            "tickers": list(dict.fromkeys(tickers)),
            "topics": [topic] if topic else [],
            "sentiment": float(overall_sent),
        }

    def pull_recent(
        self,
        query: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        since_hours: int = 6,
        region: Optional[str] = None,
        max_pages: int = 2
    ) -> List[Dict]:
        """
        免费层友好：
        - 不带 published_after（避免部分计划报 401）
        - 每页 limit<=3
        - since_hours 用本地时间阈值过滤
        """
        # —— 兜底 symbols —— #
        if not symbols:
            default_env = (getattr(settings, "MARKETAUX_DEFAULT_SYMBOLS", "") or "").strip()
            if default_env:
                symbols = [s.strip() for s in default_env.split(",") if s.strip()]
            else:
                # 从 WATCHLIST_FILE 来
                from app.utils.ticker_mapping import load_watchlist_simple
                import os
                wl_file = getattr(settings, "WATCHLIST_FILE", "")
                symbols = load_watchlist_simple(wl_file) if os.path.exists(wl_file) else []
                # 仍做一次保底（避免空 list）
                if not symbols:
                    symbols = ["AAPL","MSFT","NVDA"]

        if not symbols and not query:
            raise ValueError("marketaux needs at least `symbols` or `query` (free tier typically requires `symbols`).")

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(int(since_hours), 1))

        out: List[Dict] = []
        page = 1
        limit = min(3, int(self.page_size))

        while page <= max_pages:
            params = {
                "limit": limit,
                "language": "en",
                "filter_entities": "true",
                "include_full_content": "true",   # <— 新增
                # 如果你的账号文档写的是 include_main_text，用下面这一行替换：
                # "include_main_text": "true",
                "page": page,
            }
            if query:
                params["search"] = query
            if symbols:
                params["symbols"] = ",".join(symbols)

            data = self._request(params)
            items = data.get("data") or []
            if not items:
                break

            for it in items:
                pub = it.get("published_at") or it.get("published_at_utc") or it.get("published_at_local")
                try:
                    pub_dt = datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
                except Exception:
                    pub_dt = None
                if pub_dt and pub_dt >= cutoff:
                    out.append(self._norm_one(it))
            if len(items) < limit:
                break
            page += 1

        return out