from __future__ import annotations
from typing import List, Tuple, Literal
import math
from datetime import datetime, timezone, timedelta

from app.adapters.db.news_repo import NewsRepo, EventRepo
from app.domain.models import NewsItem, UserProfile, BehaviorEvent
from app.adapters.embeddings.hash_embedder import HashingEmbedder
from app.repositories.inmemory import InMemoryNewsRepo, InMemoryProfileRepo, InMemoryEventRepo
from app.repositories.pg_profile_repo import PgProfileRepo
from app.utils.similarity import cosine, recency_score
import numpy as np
import logging

log = logging.getLogger("app.services.news_service")
log.warning("[rec] recommend_for_user CALLED ...")

def _cos(a, b):
    import math
    if not a or not b: return 0.0
    s = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    return s/(na*nb)

def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _cos01(a: list[float] | None, b: list[float] | None) -> float:
    """余弦相似度映射到 [0,1]，空向量时返回 0。"""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0: 
        return 0.0
    na = math.sqrt(sum((a[i] or 0.0) * (a[i] or 0.0) for i in range(n))) or 1.0
    nb = math.sqrt(sum((b[i] or 0.0) * (b[i] or 0.0) for i in range(n))) or 1.0
    cos = sum((a[i] or 0.0) * (b[i] or 0.0) for i in range(n)) / (na * nb)
    # map [-1,1] -> [0,1]
    return _clip01(0.5 * (cos + 1.0))

def _recency_score(pub_dt) -> float:
    """新鲜度分（0,1]：48h 内指数衰减，最多不超过 1。"""
    if not pub_dt:
        return 0.5  # 没时间信息给个温和值
    try:
        now = datetime.now(timezone.utc)
        if getattr(pub_dt, "tzinfo", None) is not None:
            age_h = max(0.0, (now - pub_dt.astimezone(timezone.utc)).total_seconds() / 3600.0)
        else:
            # naive 当作 UTC
            age_h = max(0.0, (now.replace(tzinfo=timezone.utc) - pub_dt.replace(tzinfo=timezone.utc)).total_seconds() / 3600.0)
    except Exception:
        return 0.5
    # 48 小时半衰近似（你之前用 0.05 的 k，这里保留近似强度）
    k = 0.05
    return _clip01(math.exp(-k * age_h))

def _ensure_list(x, want_len=None, fill=0.0):
    if isinstance(x, list):
        if want_len is not None and len(x) < want_len:
            return x + [fill] * (want_len - len(x))
        return x
    return []

class NewsService:
    """
    - 负责：新闻入库→嵌入→个性化排序→行为回写画像（EMA）
    - 与文档对齐：
      * 向量对齐到用户画像空间，后续可换更强嵌入与FinBERT情绪项
      * 排序兼顾相似度与时效（文档提到重排考虑recency/quality/diversity）
      * 画像更新使用 EMA（点击/停留） 参考：u <- (1-α)*u + α*z
    """
    def __init__(self,
                 news_repo: InMemoryNewsRepo,
                 prof_repo: InMemoryProfileRepo,
                 ev_repo: InMemoryEventRepo,
                 embedder: HashingEmbedder):
        self.news_repo = news_repo
        self.prof_repo = prof_repo
        self.ev_repo = ev_repo
        self.embedder = embedder

    # ---------- 数据与嵌入 ----------
    def embed_news(self, item: NewsItem) -> NewsItem:
        text = f"{item.title}. {item.text} {' '.join(item.topics)} {' '.join(item.tickers)}"
        item.vector = self.embedder.embed_text(text)
        return item

    def ingest(self, items: List[NewsItem]):
        enriched = [self.embed_news(it) for it in items]
        self.news_repo.upsert_many(enriched)

    # ---------- 个性化排序 ----------
    def personalized_feed(self, user_id: str, limit: int = 20, strategy: str = "auto"):
        u = self.prof_repo.get_user_vectors(user_id)
        u_short = u["short"]; u_long = u["long"]; u_prof = u["prof20"]
        # 合成语义画像
        beta = 0.5
        u_sem = [beta*xs + (1-beta)*xl for xs,xl in zip(u_short, u_long)] if u_short and u_long else (u_short or u_long or [])

        # 候选：沿用你之前的“latest”
        cand = self.news_repo.latest(limit=200)

        items = []
        now = datetime.now(timezone.utc)
        for it in cand:
            s_sem  = _cos(u_sem, getattr(it, "vector", []) or [])
            s_prof = _cos(u_prof, getattr(it, "vector_prof_20d", []) or [])
            # 时间项（与现有一致或简化）
            pub = getattr(it, "published_at", None)
            age_h = 0.0
            try:
                age = (now - pub.replace(tzinfo=timezone.utc)).total_seconds()
                age_h = max(0.0, age/3600.0)
            except Exception:
                pass
            s_time = math.exp(-0.05*age_h)

            w_sem, w_prof = 0.6, 0.4
            score = (w_sem*s_sem + w_prof*s_prof) * s_time

            items.append((it, score))

        items.sort(key=lambda x: x[1], reverse=True)
        return items[:limit]
    
    # ---------- 行为反馈与画像更新（EMA） ----------
    def record_event_and_update_profile(self, ev: BehaviorEvent, alpha: float = 0.1, sentiment_weight: float = 0.2):
        self.ev_repo.add(ev)
        prof = self.prof_repo.get_or_create(ev.user_id)
        news = self.news_repo.get(ev.news_id)
        if not news or news.vector is None:
            return prof

        # z = 新闻向量 * (1 + w_sent * sentiment)
        scale = 1.0 + sentiment_weight * float(news.sentiment or 0.0)
        z = [v * scale for v in news.vector]

        # 可按事件类型调整学习率（停留>点击，踩>负向修正），这里示例化处理
        type_lr_scale = {"dwell": 1.2, "click": 1.0, "save": 1.0, "dislike": -0.8}
        a_eff = alpha * type_lr_scale.get(ev.type, 1.0)

        # EMA：u <- (1-a)*u + a*z
        u = prof.vector
        new_u = [(1 - a_eff) * u[i] + a_eff * z[i] for i in range(len(u))]
        # 归一化，避免向量发散
        import numpy as np
        norm = np.linalg.norm(new_u) + 1e-12
        prof.vector = [v / norm for v in new_u]
        self.prof_repo.save(prof)
        return prof
    
    # --- utils: 安全归一与相似度 ---
    def _l2norm(self, v):
        if not v:
            return []
        import math
        s = math.sqrt(sum(x*x for x in v)) or 1.0
        return [x/s for x in v]

    def _dot(self, a, b):
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        return float(sum(a[i]*b[i] for i in range(n)))

    def _cosine(self, a, b):
        if not a or not b:
            return 0.0
        import math
        n = min(len(a), len(b))
        na = math.sqrt(sum(a[i]*a[i] for i in range(n))) or 1.0
        nb = math.sqrt(sum(b[i]*b[i] for i in range(n))) or 1.0
        return float(sum(a[i]*b[i] for i in range(n)) / (na*nb))

    # --- 主打分：返回 [(item, score)] ---
    def score_news_for_user(self, user_id: str, candidates, w_sem: float = 0.6, w_prof: float = 0.4):
        """
        返回 [(NewsItem, score)]，且 score ∈ [0,1]。
        - 语义通道与画像通道分别映射到 [0,1] 后线性融合
        - 再乘以新鲜度分（0,1]；最终 clip 到 [0,1]
        """
        # 用户向量
        if hasattr(self.prof_repo, "get_user_vectors"):
            u = self.prof_repo.get_user_vectors(user_id)
            u_s = list(u.get("short") or [])
            u_l = list(u.get("long")  or [])
            u_p = list(u.get("prof20") or [])
        else:
            prof = self.prof_repo.get_or_create(user_id)
            u_s = list(getattr(prof, "vector", []) or [])
            u_l = [0.0]*len(u_s)
            u_p = [0.0]*20

        # 合成语义画像：short/long 组合
        if u_s and u_l:
            beta = 0.5
            u_sem = [beta*si + (1-beta)*li for si, li in zip(u_s, u_l)]
        else:
            u_sem = u_s or u_l or []

        out = []
        for it in candidates:
            news_sem  = _ensure_list(getattr(it, "vector", []) or [])
            news_prof = _ensure_list(getattr(it, "vector_prof_20d", []) or [], want_len=20)

            # 相似度（映射到 [0,1]）
            sem01  = _cos01(u_sem, news_sem)
            prof01 = _cos01(u_p,   news_prof)

            # 新鲜度
            s_time = _recency_score(getattr(it, "published_at", None))

            # 融合并 clip
            fused = (w_sem * sem01 + w_prof * prof01) * s_time
            score = _clip01(float(fused))

            out.append((it, score))

        # 降序
        out.sort(key=lambda x: x[1], reverse=True)
        return out


    # --- 推荐：可选少量实时拉取后再统一排序 ---
    def recommend_for_user(
        self,
        user_id: str,
        limit: int = 20,
        refresh: bool = False,
        mix_ratio: float = 0.2,
        symbols: list[str] | None = None,
        exclude_hours: int | None = None,
    ):
        # ---------- 新增：拿一个局部 logger，避免 self._log ----------
        import logging
        logger = logging.getLogger("app.services.news_service")

        # 1) 候选（保持原样）
        try:
            offline = self.news_repo.latest(limit=200)
        except Exception:
            offline = []

        # --- refresh 拉新：最小改动，扩大抓取量 + 明确 symbols 转发 + 强日志 ---
        if refresh:
            try:
                from app.adapters.fetchers.marketaux_fetcher import MarketauxFetcher, MarketauxConfig
                from app.services.ingest_pipeline import IngestPipeline
                from app.config import settings as cfg
                import logging
                log = logging.getLogger("app.services.news_service")

                page_size, max_pages, since_h = 3, 1, 12
                sym_used = symbols or None

                mcfg = MarketauxConfig(
                    api_key=getattr(cfg, "MARKETAUX_API_KEY", None),  # ✅ 与 debug 路径一致取 key
                    qps=float(getattr(cfg, "FETCH_QPS", 0.5)),
                    daily_budget=int(getattr(cfg, "DAILY_BUDGET_MARKETAUX", 80)),
                    page_size=page_size,
                )
                fetcher = MarketauxFetcher(mcfg)
                raw = fetcher.pull_recent(symbols=sym_used, since_hours=since_h, max_pages=max_pages)
                log.warning(f"[refresh] fetched={len(raw)} by symbols={sym_used} since_h={since_h}")

                pipe = IngestPipeline(news_repo=self.news_repo, embedder=self.embedder, watchlist=None)
                n_ing = pipe.ingest_dicts(raw)
                log.warning(f"[refresh] ingested={n_ing}")

                # 再取一次最新
                offline = self.news_repo.latest(limit=200)
                log.warning(f"[refresh] latest(after) got={len(offline)}")
            except Exception as e:
                import logging
                logging.getLogger("app.services.news_service").exception(f"[refresh] flow failed: {e}")
                # 不中断推荐，继续用 offline
        # 3) 打分（保持你的现有逻辑）
        ranked = self.score_news_for_user(user_id, offline)

        # 4) 过滤已浏览（三种事件），默认 30 天（保持你的过滤逻辑）
        since_h = int(exclude_hours) if exclude_hours is not None else 24 * 30
        seen_ids = self._get_seen_news_ids(user_id=user_id, since_hours=since_h)
        ranked_filtered = [(it, sc) for (it, sc) in ranked if getattr(it, "news_id", None) not in seen_ids]

        # 5) 整理输出 + clip
        out = []
        for it, sc in ranked_filtered[:limit]:
            out.append({
                "news_id": getattr(it, "news_id", None),
                "title": getattr(it, "title", None),
                "source": getattr(it, "source", None),
                "published_at": getattr(it, "published_at", None),
                "tickers": getattr(it, "tickers", []) or [],
                "topics": getattr(it, "topics", []) or [],
                "url": getattr(it, "url", ""),
                "score": float(_clip01(sc)),
            })
        return out

    def _get_seen_news_ids(self, user_id: str, since_hours: int = 720) -> set[str]:
        """
        统一获取“已浏览”新闻 ID 集合：
        - 优先使用 ev_repo.recent_news_ids(user_id, since_hours)
        - 否则回退到 ev_repo.all()，筛选 type in {'click','like','bookmark'} && ts >= now-since_hours
        """
        seen = set()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=int(since_hours))
        try:
            if hasattr(self.ev_repo, "recent_news_ids"):
                ids = self.ev_repo.recent_news_ids(user_id=user_id, since_hours=since_hours)
                return set(ids or [])
        except Exception as e:
            log.debug(f"[rec] ev_repo.recent_news_ids failed: {e}")

        try:
            # 回退：拉事件表
            if hasattr(self.ev_repo, "all"):
                rows = self.ev_repo.all(limit=2000)  # 适当上限，避免太大
                for r in rows or []:
                    try:
                        if str(r.get("user_id")) != str(user_id):
                            continue
                        t = (r.get("type") or "").lower()
                        if t not in {"click","like","bookmark"}:
                            continue
                        ts = r.get("ts")
                        if not ts:
                            continue
                        z = ts if getattr(ts, "tzinfo", None) is not None else ts.replace(tzinfo=timezone.utc)
                        if z >= cutoff:
                            nid = r.get("news_id")
                            if nid:
                                seen.add(str(nid))
                    except Exception:
                        continue
        except Exception as e:
            log.debug(f"[rec] ev_repo.all fallback failed: {e}")

        return seen
