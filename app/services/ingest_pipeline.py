from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional
from datetime import datetime, timezone
import hashlib
from app.domain.models import NewsItem
from app.utils.ticker_mapping import map_tickers, build_profile20_from_topics_and_signals

INDUSTRIES_11 = [
    "Utilities","Technology","Consumer Defensive","Healthcare","Basic Materials","Real Estate",
    "Energy","Industrials","Consumer Cyclical","Communication Services","Financial Services"
]
IND_INDEX = {name: i for i, name in enumerate(INDUSTRIES_11)}

def _profile9_from_sentiment(s: float) -> list[float]:
    """
    用 sentiment_score ∈ [-1,1] 规则映射到后9维（固定），范围 [0,1]。
    你可以之后微调权重函数；这里给出可解释且单调的版本。
    9维：['market_cap','growth_value','dividend','risk','liquidity','quality','valuation_safety','momentum','efficiency']
    """
    import math
    s = max(-1.0, min(1.0, float(s)))
    s_pos = max(0.0, s)          # 正情绪
    s_neg = max(0.0, -s)         # 负情绪
    s_abs = abs(s)

    market_cap       = 0.5       # 中性占位（新闻对市值偏好弱相关）
    growth_value     = s_pos     # 越正 -> 越成长
    dividend         = s_neg*0.6 # 越负 -> 越偏股息防御
    risk_tolerance   = 0.5 + 0.5*s_pos - 0.3*s_neg   # 正则更冒险，负则保守
    liquidity        = 0.6 + 0.3*s_pos               # 正则更偏流动性（题材热度）
    quality          = 0.5 + 0.4*s_pos - 0.2*s_neg   # 正→更看好质量
    valuation_safety = 0.5 + 0.4*s_neg               # 负→更看重估值安全
    momentum         = s_abs                         # 情绪强→更看重动量
    efficiency       = 0.5 + 0.3*s_pos - 0.1*s_neg   # 正→更偏效率

    out = [market_cap, growth_value, dividend, risk_tolerance, liquidity,
           quality, valuation_safety, momentum, efficiency]
    # 裁剪到 [0,1]
    return [max(0.0, min(1.0, v)) for v in out]

def build_profile20_from_topics_and_sentiment(topics: list[str], sentiment: float) -> list[float]:
    """
    20D = [11D 行业 one-hot] + [9D 投资偏好（由情感映射）]，固定值，不随用户行为变化。
    topics 里我们只取 industry 单一值（若多值取第一个）。
    """
    # 11D 独热
    onehot = [0.0]*len(INDUSTRIES_11)
    industry = None
    for t in topics or []:
        if t in INDUSTRIES_11:
            industry = t
            break
    if industry is not None:
        idx = INDUSTRIES_11.index(industry)
        onehot[idx] = 1.0
    # 9D from sentiment
    tail9 = _profile9_from_sentiment(sentiment)
    return onehot + tail9

def build_profile20_from_topics_and_signals(topics: list[str], tickers: list[str]) -> list[float]:
    """
    输出 20 维画像：
      前 11 维：行业 soft-one-hot（若多行业命中则平均分配；若无则全 0）
      后 9 维：投资偏好（此处先用 0；你也可以用 tickers/source 规则给弱信号）
    """
    v = [0.0] * 20
    # —— 11维行业 —— #
    hits = [t for t in topics if t in IND_INDEX]   # topics 已经只保留 11 行业之一，这里仍兼容多命中
    if hits:
        w = 1.0 / len(hits)
        for t in hits:
            v[IND_INDEX[t]] += w

    # —— 9维偏好 —— #
    # 这里先全部 0；后续可基于 tickers/source 长期统计进行弱信号填充。
    # v[11:20] = [0]*9
    return v

class IngestPipeline:
    """
    将“源适配器输出的字典”规范化为 NewsItem 并入库（去重、嵌入、映射）。
    约定：news_repo 提供 upsert_many(items: Iterable[NewsItem])。
    """
    def __init__(self, news_repo, embedder, watchlist: Optional[dict] = None):
        self.news_repo = news_repo
        self.embedder = embedder
        self.watchlist = watchlist or {}

    @staticmethod
    def _dedupe(items: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for it in items:
            # 优先 external_id，否则用 url hash
            ext = it.get("external_id") or hashlib.sha1((it.get("url") or "").encode("utf-8")).hexdigest()
            if ext in seen:
                continue
            seen.add(ext)
            it["external_id"] = ext
            out.append(it)
        return out
    
    # def _iso_to_dt(s: str) -> datetime:
    #     # 输入 ISO8601（可能无 tz），统一存为 UTC naive
    #     try:
    #         dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    #         if dt.tzinfo:
    #             return dt.astimezone(timezone.utc).replace(tzinfo=None)
    #         return dt
    #     except Exception:
    #         return datetime.now(timezone.utc).replace(tzinfo=None)

    # def _to_news_item(self, it: Dict) -> NewsItem:
    #     # RSS 没有 tickers：用 watchlist 轻量映射补齐
    #     tickers = it.get("tickers") or []
    #     if not tickers and self.watchlist:
    #         tickers = map_tickers(it.get("title",""), it.get("text",""), self.watchlist)

    #     # 嵌入（标题+摘要）
    #     content = (it.get("title") or "") + " " + (it.get("text") or "")
    #     vec = self.embedder.embed_text(content)

    #     return NewsItem(
    #         news_id=it["external_id"],
    #         title=it.get("title") or "",
    #         text=it.get("text") or "",
    #         source=it.get("source") or "news",
    #         url=it.get("url") or "",
    #         published_at=_iso_to_dt(it.get("published_at") or datetime.now(timezone.utc).isoformat()),
    #         tickers=tickers,
    #         topics=it.get("topics") or [],
    #         sentiment=it.get("sentiment") if it.get("sentiment") is not None else 0.0,
    #         vector=vec,
    #     )

    def _to_news_item(self, d: Dict) -> NewsItem:
        # —— 生成/选择唯一 id —— #
        nid = d.get("news_id") or d.get("external_id") or d.get("url")
        if not nid:
            nid = hashlib.sha1((d.get("title","") + d.get("url","") + d.get("text","")).encode("utf-8")).hexdigest()

        # —— 解析时间 —— #
        pub = d.get("published_at")
        if isinstance(pub, str):
            try:
                pub = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            except Exception:
                pub = None

        # —— 关键：保留 tickers / topics / sentiment —— #
        tickers = d.get("tickers") or []
        if isinstance(tickers, list):
            tickers = [t for t in tickers if isinstance(t, str) and t.strip()]

        topics = d.get("topics") or []
        if isinstance(topics, list):
            topics = [t for t in topics if isinstance(t, str) and t.strip()]

        try:
            sentiment = float(d.get("sentiment") or 0.0)
        except Exception:
            sentiment = 0.0

        item = NewsItem(
            news_id=nid,
            title=d.get("title") or "",
            text=d.get("text") or "",
            source=d.get("source") or "",
            url=d.get("url") or "",
            published_at=pub,
            tickers=tickers,
            topics=topics,
            sentiment=sentiment,
            vector=[],   # 先占位，下面再填
        )

        # —— 向量 —— #
        content = f"{item.title}\n{item.text}".strip()
        item.vector = self.embedder.embed_text(content)

        # # —— 新增：构造 20维画像向量（写到 item.extra，供 repo 入库）——
        # prof20 = build_profile20_from_topics_and_signals(item.topics, item.tickers)
        # # 为了最小侵入，不改 NewsItem；把扩展字段放在临时属性上让 repo 使用
        # setattr(item, "_profile20", prof20)
        
        prof20 = build_profile20_from_topics_and_sentiment(item.topics, sentiment)
        setattr(item, "_profile20", prof20)
        return item
    
    def ingest_dicts(self, raw_items: List[Dict]) -> Tuple[int, int]:
        """返回 (去重后数量, 实际入库数量)"""
        clean = self._dedupe(raw_items)
        objs = [self._to_news_item(it) for it in clean]
        # self.news_repo.upsert_many(objs)
        # return len(clean), len(objs)
        res = self.news_repo.upsert_many(objs)
        # stored = 新插入 + 修改
        stored = int(res.get("nUpserted", 0)) + int(res.get("nModified", 0))
        return len(clean), stored