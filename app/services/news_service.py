from __future__ import annotations
from typing import List, Tuple, Literal
import math
from datetime import datetime, timezone
from app.domain.models import NewsItem, UserProfile, BehaviorEvent
from app.repositories.inmemory import InMemoryNewsRepo, InMemoryProfileRepo, InMemoryEventRepo
from app.adapters.embeddings.hash_embedder import HashingEmbedder
from app.utils.similarity import cosine, recency_score
import numpy as np

def _cos(a, b):
    import math
    if not a or not b: return 0.0
    s = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    return s/(na*nb)

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
    # def personalized_feed(self, 
    #                       user_id: str, 
    #                       limit: int = 20,
    #                       strategy: Literal["auto", "personalized", "trending"] = "auto",
    #                       w_sim: float = 0.7,
    #                       w_time: float = 0.3) -> List[Tuple[NewsItem, float]]:
    #     """
    #     返回 [(news_item, score)] 列表。
    #     strategy:
    #       - "auto": 有画像则个性化，否则走新鲜度
    #       - "personalized": 强制个性化（即使画像很弱）
    #       - "trending": 仅按新鲜度（冷启动兜底/热门）
    #     """
    #     profile = self.prof_repo.get_or_create(user_id)
    #     items = self.news_repo.latest(limit=200)  # 拉一个较大的候选池再排序
        
    #     # 画像强度（范数）用于冷启动判断
    #     p_vec = np.asarray(profile.vector, dtype=float)
    #     p_norm = float(np.linalg.norm(p_vec))
    #     cold = p_norm < 1e-6

    #     scored = []
    #     for it in items:
    #         if strategy == "trending" or (strategy == "auto" and cold):
    #             # 冷启动/热门：只看新鲜度
    #             s_time = recency_score(it.published_at)
    #             score = s_time
    #         else:
    #             if it.vector is None:
    #                 it = self.embed_news(it)
    #                 assert len(profile.vector) == len(it.vector) == getattr(self.embedder, "dim", 32), \
    #    f"dim mismatch: profile={len(profile.vector)}, news={len(it.vector)}, embedder={getattr(self.embedder,'dim',None)}"
    #             # 个性化：相似度 × 新鲜度 融合
    #             s_sim = cosine(profile.vector, it.vector)
    #             s_time = recency_score(it.published_at)
    #             score = w_sim * s_sim + w_time * s_time
    #         scored.append((it, float(score)))
    #     scored.sort(key=lambda x: x[1], reverse=True)
    #     return scored[:limit]

    # 读用户画像
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
        candidates: List[NewsItem]（通常来自 MongoNewsRepo.latest()）
        打分融合：
        - 语义通道：user_sem = 0.6*short + 0.4*long，再与 news.vector 做点积/余弦
        - 画像通道：user_profile_20d 与 news.vector_prof_20d 做余弦
        - 简单新鲜度奖励 freshness_bonus（最新的略加分，避免一水老文）
        """
        # 取用户三路向量
        if hasattr(self.prof_repo, "get_user_vectors"):
            u = self.prof_repo.get_user_vectors(user_id)
            u_s = list(u.get("short") or [])
            u_l = list(u.get("long") or [])
            u_p = list(u.get("prof20") or [])
        else:
            # 兼容 MongoProfileRepo 旧式仅一条向量
            prof = self.prof_repo.get_or_create(user_id)
            u_s = list(getattr(prof, "vector", []) or [])
            u_l = [0.0]*len(u_s)
            u_p = [0.0]*20

        # 组合短长通道 → user_sem
        user_sem = [0.6*si + 0.4*li for si, li in zip(self._l2norm(u_s), self._l2norm(u_l))] or []
        user_prof = u_p

        # 计算新鲜度窗口
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        horizon = timedelta(hours=48)  # 48小时窗口内线性衰减
        scored = []
        for it in candidates:
            news_sem = list(getattr(it, "vector", []) or [])
            news_prof = list(getattr(it, "vector_prof_20d", []) or [])
            # sim
            sem_sim = self._dot(user_sem, self._l2norm(news_sem))   # SRP向量已单位化，点积≈余弦
            prof_sim = self._cosine(user_prof, news_prof)
            score = w_sem*sem_sim + w_prof*prof_sim

            # 新鲜度奖励（最多 +0.1）
            pub = getattr(it, "published_at", None)
            bonus = 0.0
            if pub:
                try:
                    # pub 可能是 naive 或 tz-aware，统一当作 UTC naive 使用
                    if getattr(pub, "tzinfo", None) is not None:
                        pub = pub.astimezone(timezone.utc).replace(tzinfo=None)
                    age = (now.replace(tzinfo=None) - pub)
                    if age <= horizon:
                        bonus = 0.1 * max(0.0, 1.0 - (age / horizon))
                except Exception:
                    pass

            scored.append((it, float(score + bonus)))

        # 分数降序
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # --- 推荐：可选少量实时拉取后再统一排序 ---
    # def recommend_for_user(self, user_id: str, limit: int = 20, refresh: bool = False, mix_ratio: float = 0.2, symbols: list[str] | None = None):
    #     """
    #     - 默认只用库里候选集：latest(200)
    #     - refresh=True 时：先临时拉一小撮(≈limit*mix_ratio)，入库后与库里候选合并再排序
    #     """
    #     # 1) 候选：离线
    #     offline = self.news_repo.latest(limit=200)

    #     # 2) 可选：实时拉取一小撮并入库
    #     if refresh:
    #         try:
    #             from app.adapters.fetchers.marketaux_fetcher import MarketauxFetcher, MarketauxConfig
    #             from app.services.ingest_pipeline import IngestPipeline
    #             from app.config import settings
    #             mcfg = MarketauxConfig(
    #                 api_key=settings.MARKETAUX_API_KEY,
    #                 qps=float(getattr(settings, "FETCH_QPS", 0.5)),
    #                 daily_budget=int(getattr(settings, "DAILY_BUDGET_MARKETAUX", 80)),
    #                 page_size=3,  # 免费层友好
    #             )
    #             mfetch = MarketauxFetcher(mcfg)
    #             # 如果没传 symbols，按你之前的兜底（us/in 或 env）
    #             raw = mfetch.pull_recent(symbols=symbols or None, since_hours=6, max_pages=1)
    #             pipe = IngestPipeline(news_repo=self.news_repo, embedder=self.embedder, watchlist=None)
    #             pipe.ingest_dicts(raw)
    #             # 再取一遍最新候选（让刚入库的也进来）
    #             offline = self.news_repo.latest(limit=200)
    #         except Exception:
    #             # 有配额/网络问题时不要影响流程
    #             pass

    #     # 3) 打分排序
    #     ranked = self.score_news_for_user(user_id, offline)
    #     # 4) 整理返回
    #     out = []
    #     for it, sc in ranked[:limit]:
    #         out.append({
    #             "news_id": it.news_id,
    #             "title": it.title,
    #             "source": it.source,
    #             "published_at": getattr(it, "published_at", None),
    #             "tickers": getattr(it, "tickers", []) or [],
    #             "topics": getattr(it, "topics", []) or [],
    #             "url": getattr(it, "url", ""),
    #             "score": sc,
    #         })
    #     return out


    def recommend_for_user(
        self,
        user_id: str,
        limit: int = 20,
        refresh: bool = False,
        mix_ratio: float = 0.2,
        symbols: list[str] | None = None,
        exclude_hours: int | None = None,   # ✅ 新增：过滤最近交互的时间窗（小时）
    ):
        """
        - 默认只用库里候选集：latest(200)
        - refresh=True 时：先临时拉一小撮(≈limit*mix_ratio)，入库后与库里候选合并再排序
        - 返回前会过滤掉最近交互过的新闻（时间窗默认 settings.RECENT_EXCLUDE_HOURS）
        """
        from app.config import settings

        # 1) 候选：离线
        offline = self.news_repo.latest(limit=200)

        # 2) 可选：实时拉取一小撮并入库
        if refresh:
            try:
                from app.adapters.fetchers.marketaux_fetcher import MarketauxFetcher, MarketauxConfig
                from app.services.ingest_pipeline import IngestPipeline
                mcfg = MarketauxConfig(
                    api_key=settings.MARKETAUX_API_KEY,
                    qps=float(getattr(settings, "FETCH_QPS", 0.5)),
                    daily_budget=int(getattr(settings, "DAILY_BUDGET_MARKETAUX", 80)),
                    page_size=3,  # 免费层友好
                )
                mfetch = MarketauxFetcher(mcfg)
                raw = mfetch.pull_recent(symbols=symbols or None, since_hours=6, max_pages=1)
                pipe = IngestPipeline(news_repo=self.news_repo, embedder=self.embedder, watchlist=None)
                pipe.ingest_dicts(raw)
                # 再取一遍最新候选（让刚入库的也进来）
                offline = self.news_repo.latest(limit=200)
            except Exception:
                # 有配额/网络问题时不要影响流程
                pass

        # 3) 打分排序（得到 [(NewsItem, score), ...]）
        ranked = self.score_news_for_user(user_id, offline)

        # 3.5) ✅ 过滤最近交互
        ex_hours = int(exclude_hours if exclude_hours is not None else getattr(settings, "RECENT_EXCLUDE_HOURS", 72))
        try:
            recent_ids = self.ev_repo.recent_news_ids(user_id=user_id, since_hours=ex_hours)
        except Exception:
            recent_ids = set()

        ranked_filtered = [(it, sc) for (it, sc) in ranked if it.news_id not in recent_ids]

        # 若过滤后不足 limit，可用“最近交互过”的做回填（避免空列表）
        if len(ranked_filtered) < limit:
            fallback = [(it, sc) for (it, sc) in ranked if it.news_id in recent_ids]
            ranked_filtered.extend(fallback)

        # 4) 整理返回
        out = []
        for it, sc in ranked_filtered[:limit]:
            out.append({
                "news_id": it.news_id,
                "title": it.title,
                "source": it.source,
                "published_at": getattr(it, "published_at", None),
                "tickers": getattr(it, "tickers", []) or [],
                "topics": getattr(it, "topics", []) or [],
                "url": getattr(it, "url", ""),
                "score": sc,
            })
        return out