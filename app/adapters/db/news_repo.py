from typing import Optional, List
from datetime import datetime

from app.adapters.db.database_client import get_mongo_db
from app.model.models import NewsItem 
from pymongo import UpdateOne, MongoClient

import logging
log = logging.getLogger("app.adapters.db.news_repo")
  
class NewsRepo:
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

    def upsert_many_dicts(self, items: list[dict]) -> dict:
        """
        批量 upsert（dict 输入）。只做日志增强，不改变你原逻辑。
        """
        try:
            total = len(items or [])
            log.warning(f"[MongoNewsRepo] upsert_many_dicts: incoming={total}")

            if not items:
                return {"nInserted": 0, "nUpserted": 0, "nMatched": 0, "nModified": 0}

            ops = []
            now = datetime.utcnow()

            miss_id = 0
            keys_sample = set()

            for it in items:
                if isinstance(it, dict):
                    keys_sample.update(list(it.keys())[:20])

                news_id = (it.get("news_id") or "").strip()
                if not news_id:
                    miss_id += 1
                    continue

                title        = it.get("title", "") or ""
                text         = it.get("text", "") or ""
                source       = it.get("source", "") or ""
                url          = it.get("url", "") or it.get("link", "") or ""

                published_at = it.get("published_at") or now
                # 尽量把字符串时间转成 datetime
                if isinstance(published_at, str):
                    try:
                        # 允许 'Z'
                        published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
                        published_at = now

                vector       = list(it.get("vector") or [])
                tickers      = list(it.get("tickers") or [])
                topics       = list(it.get("topics") or [])

                try:
                    sentiment = float(it.get("sentiment", 0.0) or 0.0)
                except Exception:
                    sentiment = 0.0

                # 画像 20 维
                prof20 = it.get("_profile20", None)
                if prof20 is None:
                    prof20 = it.get("vector_prof_20d", None)

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
                        "vector_prof_20d": prof20,
                        "tickers": tickers,
                        "topics": topics,
                        "sentiment": sentiment,
                        "updated_at": now,
                    }
                }
                ops.append(UpdateOne({"news_id": news_id}, update, upsert=True))

            log.warning(f"[MongoNewsRepo] upsert_many_dicts: ops={len(ops)}, missing_news_id={miss_id}, "
                        f"keys_sample={sorted(list(keys_sample))[:20]}")

            if not ops:
                return {"nInserted": 0, "nUpserted": 0, "nMatched": 0, "nModified": 0}

            res = self.col.bulk_write(ops, ordered=False)

            out = {
                "nInserted": getattr(res, "inserted_count", 0),
                "nUpserted": len(getattr(res, "upserted_ids", []) or []),
                "nMatched": getattr(res, "matched_count", 0),
                "nModified": getattr(res, "modified_count", 0),
            }
            log.warning(f"[MongoNewsRepo] bulk_write result: {out}")
            return out

        except Exception as e:
            log.exception(f"[MongoNewsRepo] upsert_many_dicts ERROR: {e}")
            raise

    def raw_latest_docs(self, limit: int = 20) -> list[dict]:
        proj = {
            "_id": 0,
            "news_id": 1,
            "title": 1,
            "source": 1,
            "url": 1,
            "published_at": 1,
            "tickers": 1,
            "topics": 1,
            "vector": 1,
            "vector_prof_20d": 1,
            "sentiment": 1,
        }
        cur = self.col.find({}, proj).sort("published_at", -1).limit(int(limit))
        docs = list(cur)
        return docs

    def count_all(self) -> int:
        try:
            return int(self.col.estimated_document_count())
        except Exception:
            return int(self.col.count_documents({}))

# === 新增：MongoEventRepo（与 NewsRepo 同风格；同步 PyMongo；使用远端 Mongo）===

from datetime import datetime, timezone, timedelta
from typing import Iterable, Optional, Set

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

class EventRepo:
    """
    事件仓库（click / like / bookmark）：
    - 同步 PyMongo，与你的 NewsRepo 风格一致
    - 使用同一个远端 Mongo（由 main 的 SSH 隧道 + 你的 Mongo 客户端配置保证）
    - 只实现你当前用到的方法：add() / recent_news_ids() / ping()
    """
    # def __init__(self, uri: str = None, db_name: str = "finsight", col_name: str = "events"):
    #     # 和 NewsRepo 一样的用法：如果你的 NewsRepo 是通过 get_mongo_db() 获取连接，
    #     # 那就改成用同样的方式拿 client / collection。
    #     #
    #     # 如果你当前的 NewsRepo 是:
    #     #   self.client = MongoClient(uri); self.col = self.client[db][col]
    #     # 那这里也保持一致：
    #     self.client = MongoClient(uri) if uri else MongoClient()
    #     self.col = self.client[db_name][col_name]
    #     self.ensure_indexes()

    #     # 你现在的 user_router 是同步调用 ev_repo.add(...)，所以这里做同步 add
   
    # def add(self, ev) -> None:
    #     """
    #     ev: BehaviorEvent 或兼容 dict，字段至少包含：
    #         user_id, news_id, type, ts (缺省则用 now UTC)
    #     """
    #     d = ev.model_dump() if hasattr(ev, "model_dump") else dict(ev)
    #     if not d.get("ts"):
    #         d["ts"] = datetime.now(timezone.utc)
    #     try:
    #         self.col.insert_one({
    #             "user_id": d.get("user_id"),
    #             "news_id": d.get("news_id"),
    #             "type":    d.get("type"),     # "click" | "like" | "bookmark" | ...
    #             "ts":      d.get("ts"),
    #             # 你现在没用到其它字段（比如 dwell_ms / liked / bookmarked），先不展开
    #         })
    #     except PyMongoError as e:
    #         # 只打日志，不抛异常以不影响主流程
    #         import logging
    #         logging.getLogger("app.mongo.events").warning(f"[MongoEventRepo] add failed: {e}")

    def __init__(self, uri: str, db_name: str = "finsight", col_name: str = "events"):
        self.client = MongoClient(uri)
        self.col = self.client[db_name][col_name]
        self.tog = self.client[db_name].get_collection("user_event_toggles")
        self.col.create_index([("user_id", 1), ("news_id", 1), ("type", 1), ("ts", -1)])
        self.tog.create_index([("user_id", 1), ("news_id", 1), ("action", 1)], unique=True)

    def add(self, ev: dict):
        ev = dict(ev or {})
        ev.setdefault("ts", datetime.utcnow())
        self.col.insert_one(ev)
        return True

    def get_toggle_state(self, user_id: str, news_id: str, action: str) -> bool:
        doc = self.tog.find_one({"user_id": user_id, "news_id": news_id, "action": action},
                                {"_id": 0, "on": 1})
        return bool(doc.get("on")) if doc else False

    # ⬇️ 新增：读/写“快照”
    def get_toggle_snapshot(self, user_id: str, news_id: str, action: str) -> list[float] | None:
        doc = self.tog.find_one(
            {"user_id": user_id, "news_id": news_id, "action": action},
            {"_id": 0, "prev_prof20": 1}
        )
        snap = (doc or {}).get("prev_prof20")
        # 只有当是“长度==20 的 list[float]”时才认为有效快照；否则返回 None
        if isinstance(snap, list) and len(snap) == 20:
            try:
                return [float(x) for x in snap]
            except Exception:
                return None
        return None

    def set_toggle_state(self, user_id: str, news_id: str, action: str,
                        on: bool, prev_prof20: list[float] | None = ...):
        """
        - prev_prof20=None  -> 用 $unset 清除快照
        - prev_prof20=省略  -> 只改 'on'
        - prev_prof20=list  -> $set 这个快照
        """
        q = {"user_id": user_id, "news_id": news_id, "action": action}
        base_set = {"on": bool(on), "updated_at": datetime.utcnow()}

        if prev_prof20 is ...:  # 不处理快照，仅更新 on
            self.tog.update_one(q, {"$set": base_set}, upsert=True)
            return

        if prev_prof20 is None:
            self.tog.update_one(q, {"$set": base_set, "$unset": {"prev_prof20": ""}}, upsert=True)
            return

        # list 情况：规范化成 20 维
        v = list(prev_prof20)
        if len(v) < 20: v += [0.0] * (20 - len(v))
        v = v[:20]
        self.tog.update_one(q, {"$set": {**base_set, "prev_prof20": v}}, upsert=True)

    def exists_by_type(self, user_id: str, news_id: str, type: str) -> bool:
        """
        返回是否“该用户对该新闻的同类型行为当前处于已开启状态（on）或最近一次是 add”。
        - 首选读取 toggle 快照表（user_event_toggles）里的 on 状态；
        - 若没有 toggle 记录，则回落到 events 表，取最新一条同类型事件，看 op 是否 'add'。
        仅对同类型（like/save）做去重，不会被 click 影响。
        """
        action = type  # 我们统一：type 'like'/'save' 对应 toggle.action 'like'/'save'

        # 1) 先看 toggle 状态（如果存在，最权威）
        tog = self.tog.find_one(
            {"user_id": user_id, "news_id": news_id, "action": action},
            {"_id": 0, "on": 1},
        )
        if tog is not None:
            return bool(tog.get("on", False))

        # 2) 回落：查 events 集合里的最新一条同类型事件
        doc = self.col.find_one(
            {"user_id": user_id, "news_id": news_id, "type": type},
            sort=[("ts", DESCENDING)],
            projection={"_id": 0, "op": 1},
        )
        if doc is None:
            return False
        # 如果最近一次是 add，就视为存在；remove 则视为不存在
        return (doc.get("op") == "add")
        
    def ensure_indexes(self):
        # 按你 filters 的使用场景建必要索引
        self.col.create_index([("user_id", ASCENDING), ("ts", DESCENDING)])
        self.col.create_index([("news_id", ASCENDING), ("ts", DESCENDING)])
        self.col.create_index([("type", ASCENDING), ("ts", DESCENDING)])

    def recent_news_ids(self, user_id: str, since_hours: int = 72) -> Set[str]:
        """
        返回用户最近 since_hours 小时内发生过 3 种行为（click/like/save）的 news_id 集合。
        用于推荐时过滤“看过/互动过”的内容。
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=int(since_hours))
        cur = self.col.find(
            {
                "user_id": user_id,
                "type": {"$in": ["click", "like", "save"]},
                "ts": {"$gte": cutoff},
            },
            {"_id": 0, "news_id": 1},
        ).sort("ts", DESCENDING)
        return {doc.get("news_id") for doc in cur if doc.get("news_id")}

    def ping_detail(self) -> tuple[bool, Optional[str]]:
        try:
            self.client.admin.command("ping")
            return True, None
        except Exception as e:
            return False, f"{e.__class__.__name__}: {e}"

    def ping(self) -> bool:
        ok, _ = self.ping_detail()
        return ok

