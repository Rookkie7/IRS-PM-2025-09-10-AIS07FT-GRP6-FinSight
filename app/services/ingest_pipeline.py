from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional
from datetime import datetime, timezone
import hashlib
from app.domain.models import NewsItem
from app.utils.ticker_mapping import map_tickers, build_profile20_from_topics_and_signals

import logging
log = logging.getLogger("app.ingest")

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
        
        prof20 = build_profile20_from_topics_and_sentiment(item.topics, sentiment)
        setattr(item, "_profile20", prof20)
        return item
    
    def ingest_dicts(self, raw_list: list[dict]) -> int:
        import logging
        log = logging.getLogger("app.services.ingest_pipeline")
        if not raw_list:
            log.warning("[ingest] empty raw_list")
            return 0

        clean = self._dedupe(raw_list)
        items = []
        for d in clean:
            try:
                it = self._to_news_item(d)  # ✅ 会计算 _profile20（情绪版）
                items.append(it)
            except Exception as e:
                log.warning(f"[ingest] skip one: {e}")

        log.warning(f"[ingest] to-upsert items={len(items)}, sample_keys={list(clean[0].keys()) if clean else []}")

        # ✅ 老路径：直接 upsert_many(NewsItem)
        try:
            res = self.news_repo.upsert_many(items)
            written = int(res.get("nUpserted", 0)) + int(res.get("nModified", 0))
        except AttributeError:
            # 兜底：某些实现只有 dict 版，则把 _profile20 显式塞到 vector_prof_20d 再调用
            as_dicts = []
            for it in items:
                dd = it.model_dump()
                prof20 = getattr(it, "_profile20", None) or getattr(it, "vector_prof_20d", None) or []
                if prof20:
                    dd["vector_prof_20d"] = list(prof20)
                as_dicts.append(dd)
            res = self.news_repo.upsert_many_dicts(as_dicts)
            written = int(res.get("nUpserted", 0)) + int(res.get("nModified", 0))

        log.warning(f"[ingest] upsert result: {res}")

        try:
            n = self.news_repo.count_all()
            latest = getattr(self.news_repo, "raw_latest_docs", lambda n: [])(limit=3)
            log.warning(f"[ingest] after-write: count_all={n}, latest_titles={[x.get('title') for x in latest]}")
        except Exception as e:
            log.exception(f"[ingest] after-write check failed: {e}")

        return written