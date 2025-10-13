from typing import Optional, List
from bson import ObjectId
from app.adapters.db.mongo_client import get_db
from app.model.models import News

class NewsRepoMongo:
    def __init__(self):
       self.col = get_db()["news"]

    async def get_many(self, ids: List[str]) -> List[dict]:
        obj_ids, str_ids = [], []
        for x in ids:
            try:
                obj_ids.append(ObjectId(x))
            except Exception:
                str_ids.append(x)
        q = {"$or": []}
        if obj_ids: q["$or"].append({"_id": {"$in": obj_ids}})
        if str_ids: q["$or"].append({"_id": {"$in": str_ids}})
        if not q["$or"]:
            return []
        cursor = self.col.find(q)
        return [doc async for doc in cursor]