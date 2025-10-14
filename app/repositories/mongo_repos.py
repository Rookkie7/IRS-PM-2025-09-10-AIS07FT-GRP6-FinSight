from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
from pymongo import UpdateOne  
from datetime import datetime, timezone, timedelta 
from pymongo import MongoClient, ASCENDING
from app.domain.models import NewsItem, BehaviorEvent, UserProfile


# def _to_utc_naive(dt: datetime | None) -> datetime | None:
#     if dt is None:
#         return None
#     if dt.tzinfo is None:
#         return dt
#     return dt.astimezone(timezone.utc).replace(tzinfo=None)

class MongoNewsRepo:
    # def __init__(self, mongo_uri: str, db_name: str = "finsight"):
    #     self.client = MongoClient(mongo_uri)
    #     self.col = self.client[db_name]["news"]
    #     self.col.create_index([("news_id", ASCENDING)], unique=True)
    #     self.col.create_index([("published_at", ASCENDING)])

    def __init__(self, uri: str, db_name: str = "finsight", col_name: str = "news"):
        self.client = MongoClient(uri)
        self.col = self.client[db_name][col_name]
        self.col.create_index("news_id", unique=True)
        self.col.create_index([("published_at", -1)])
                                 
    def clear(self):
        self.col.delete_many({})
    
    def ping_detail(self) -> tuple[bool, str | None]:
        try:
            self.client.admin.command("ping")
            return True, None
        except Exception as e:
            return False, f"{e.__class__.__name__}: {e}"

    def ping(self) -> bool:
        ok, _ = self.ping_detail()
        return ok
        
    # def upsert_many(self, items: Iterable[NewsItem]):
    #     print("[MongoNewsRepo] upsert_many using", "bulk" or "single")  # 按你选择写

    #     ops: list[UpdateOne] = []
    #     for it in items:
    #         d = it.model_dump()
    #         d["published_at"] = _to_utc_naive(d.get("published_at"))
    #         ops.append(
    #             UpdateOne({"news_id": d["news_id"]}, {"$set": d}, upsert=True)
    #         )
    #     if ops:
    #         self.col.bulk_write(ops)

    def upsert_many(self, items: List[NewsItem]):
        if not items:
            return {"nInserted": 0, "nUpserted": 0, "nMatched": 0, "nModified": 0}

        ops = []
        now = datetime.utcnow()

        for it in items:
            news_id      = getattr(it, "news_id", None)
            if not news_id:
                # 跳过无 news_id 的脏数据
                continue

            title        = getattr(it, "title", "") or ""
            text         = getattr(it, "text", "") or ""
            source       = getattr(it, "source", "") or ""
            url          = getattr(it, "url", "") or getattr(it, "link", "") or ""
            published_at = getattr(it, "published_at", None) or now
            vector       = list(getattr(it, "vector", []) or [])
            # vector_prof_20d = list(getattr(it, "vector_prof_20d", []) or [])
            tickers      = list(getattr(it, "tickers", []) or [])
            topics       = list(getattr(it, "topics", []) or [])
            try:
                sentiment = float(getattr(it, "sentiment", 0.0) or 0.0)
            except Exception:
                sentiment = 0.0

            # === 新增：读取 20 维画像向量 ===
            # 优先使用 pipeline 放进来的临时属性 _profile20；次选对象上已有的 vector_prof_20d（以便兼容未来模型字段）
            prof20 = getattr(it, "_profile20", None)
            if prof20 is None:
                prof20 = getattr(it, "vector_prof_20d", None)

            # 规范化为 list[float]，并限制长度到 20（多余截断，缺失补零）
            if not isinstance(prof20, list):
                prof20 = []
            try:
                prof20 = [float(x) for x in prof20]
            except Exception:
                prof20 = []
            if len(prof20) < 20:
                prof20 = prof20 + [0.0] * (20 - len(prof20))
            elif len(prof20) > 20:
                prof20 = prof20[:20]

            # ✅ news_id 只放在 $setOnInsert，不要放进 $set
            update = {
                "$setOnInsert": {
                    "news_id": news_id,
                    "created_at": now,
                },
                "$set": {
                    "title": title,
                    "text": text,
                    "source": source,
                    "url": url,
                    "published_at": published_at,
                    "vector": vector,
                    "vector_prof_20d": prof20,  # 新增：20维画像向量
                    "tickers": tickers,
                    "topics": topics,
                    "sentiment": sentiment,
                    "updated_at": now,
                }
            }

            ops.append(UpdateOne(
                {"news_id": news_id},
                update,
                upsert=True
            ))

        if not ops:
            return {"nInserted": 0, "nUpserted": 0, "nMatched": 0, "nModified": 0}

        res = self.col.bulk_write(ops, ordered=False)
        # 统一返回一些统计值，便于上层展示
        return {
            "nInserted": getattr(res, "inserted_count", 0),
            "nUpserted": len(getattr(res, "upserted_ids", []) or []),
            "nMatched": getattr(res, "matched_count", 0),
            "nModified": getattr(res, "modified_count", 0),
        }
    
    def all(self) -> List[NewsItem]:
        return [NewsItem(**doc) for doc in self.col.find().sort("published_at", -1)]

    def get(self, news_id: str) -> Optional[NewsItem]:
        doc = self.col.find_one({"news_id": news_id})
        return NewsItem(**doc) if doc else None

    def latest(self, limit: int = 200) -> List[NewsItem]:
        """
        返回按发布时间倒序的最近新闻，转换为 NewsItem 模型。
        """
        cursor = (
            self.col.find({})
            .sort("published_at", -1)
            .limit(int(limit))
        )
        items: List[NewsItem] = []
        for doc in cursor:
            # published_at 如果是带 tz 的，存库时我们已去 tz；这里按原样塞回去
            items.append(NewsItem(
                news_id=doc["news_id"],
                title=doc.get("title", ""),
                text=doc.get("text", ""),
                source=doc.get("source", ""),
                url=doc.get("url", ""),
                published_at=doc.get("published_at"),
                tickers=doc.get("tickers", []),
                topics=doc.get("topics", []),
                sentiment=doc.get("sentiment", 0.0),
                vector=doc.get("vector", []),
                vector_prof_20d=doc.get("vector_prof_20d", []),
            ))
        return items
    
    def raw_latest_docs(self, limit: int = 50) -> list[dict]:
        """
        仅供调试使用：直接从 Mongo 读出原始文档所需字段，
        避免通过 NewsItem 模型转换导致的字段丢失/置空。
        """
        proj = {
            "_id": 0,
            "news_id": 1,
            "title": 1,
            "source": 1,
            "url": 1,
            "published_at": 1,
            "tickers": 1,
            "topics": 1,
            "sentiment": 1,
            "text": 1,                # 真实文本
            "vector": 1,              # 64d 语义向量
            "vector_prof_20d": 1,     # 20d 画像向量
        }
        cur = (
            self.col.find({}, proj)
            .sort("published_at", -1)
            .limit(int(limit))
        )
        return list(cur)

class MongoEventRepo:
    def __init__(self, mongo_uri: str, db_name: str = "finsight"):
        self.client = MongoClient(mongo_uri)
        self.col = self.client[db_name]["events"]
        self.col.create_index([("user_id", ASCENDING)])
        self.col.create_index([("ts", ASCENDING)])

    def ping_detail(self) -> tuple[bool, str | None]:
        try:
            self.client.admin.command("ping")
            return True, None
        except Exception as e:
            return False, f"{e.__class__.__name__}: {e}"

    def ping(self) -> bool:
        ok, _ = self.ping_detail()
        return ok
        
    def add(self, ev: BehaviorEvent):
        d = ev.model_dump()
        if d.get("ts") is not None:
            d["ts"] = d["ts"].astimezone(timezone.utc).replace(tzinfo=None)
        self.col.insert_one(d)

    def all(self) -> List[BehaviorEvent]:
        return [BehaviorEvent(**doc) for doc in self.col.find().sort("ts", -1)]

    def recent_interacted_news_ids(self, user_id: str, since_hours: int = 72) -> set[str]:
        """
        返回用户最近 since_hours 小时内有过交互的 news_id 集合（click / like / bookmark）。
        你的 BehaviorEvent 里如果有 liked/bookmarked 字段，这里一并视作“交互”。
        """
        since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        q = {"user_id": user_id, "ts": {"$gte": since}}
        # 如果你的事件集合里字段名不同（如 dwell_ms / liked / bookmarked），按需调整投影
        cur = self.col.find(q, {"_id": 0, "news_id": 1})
        return {d.get("news_id") for d in cur if d.get("news_id")}
    
    def recent_news_ids(self, user_id: str, since_hours: int = 72) -> set[str]:
        """返回指定时间窗内此用户交互过的 news_id 集合。"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=int(since_hours))
        cur = self.col.find(
            {"user_id": user_id, "ts": {"$gte": cutoff}},
            {"_id": 0, "news_id": 1}
        )
        return {d.get("news_id") for d in cur if d.get("news_id")}

class MongoProfileRepo:
    """把用户画像也存 Mongo（用于 VECTOR_BACKEND=mongo）"""
    def __init__(self, mongo_uri: str, db_name: str = "finsight", dim: int = 32):
        self.client = MongoClient(mongo_uri)
        self.col = self.client[db_name]["user_profiles"]
        self.col.create_index([("user_id", ASCENDING)], unique=True)
        self.dim = dim

    def ping_detail(self) -> tuple[bool, str | None]:
        try:
            self.client.admin.command("ping")
            return True, None
        except Exception as e:
            return False, f"{e.__class__.__name__}: {e}"

    def ping(self) -> bool:
        ok, _ = self.ping_detail()
        return ok
        
    def get_or_create(self, user_id: str) -> UserProfile:
        doc = self.col.find_one({"user_id": user_id})
        if doc:
            return UserProfile(user_id=doc["user_id"], vector=list(doc.get("vector", [0.0]*self.dim)))
        prof = UserProfile(user_id=user_id, vector=[0.0]*self.dim)
        self.col.insert_one({"user_id": prof.user_id, "vector": prof.vector})
        return prof

    def save(self, prof: UserProfile):
        self.col.update_one(
            {"user_id": prof.user_id},
            {"$set": {"vector": list(prof.vector)}},
            upsert=True
        )
