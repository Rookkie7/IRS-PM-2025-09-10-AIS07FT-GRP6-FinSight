# -*- coding: utf-8 -*-
from typing import List, Tuple, Optional, Any
from bson import ObjectId
from app.ports.vector_index import VectorIndexPort
from app.adapters.db.mongo_client import get_db
from app.model.models import EmbeddingVector

class MongoVectorIndex(VectorIndexPort):
    """
    依赖 MongoDB 的 $vectorSearch（Atlas/本地支持 Search+Vector 的版本）
    需要预先在集合上创建 vector 索引:
      index name: embedding_idx
      fields.path: "embedding.values"
      numDimensions: 32
      similarity: "cosine" (或 dotProduct)
    """
    def __init__(self,
                 collection_name: str,
                 index_name: str = "embedding_idx",
                 path: str = "embedding.values"):
        self.col = get_db()[collection_name]
        self.index_name = index_name
        self.path = path

    async def upsert(self, doc_id: str, vector: EmbeddingVector, metadata: dict) -> None:
        _id: Any = self._to_id(doc_id)
        await self.col.update_one(
            {"_id": _id},
            {"$set": {"embedding": vector.dict(), **metadata}},
            upsert=True
        )

    async def search(self, vector: EmbeddingVector, top_k: int = 10,
                     filters: Optional[dict] = None) -> List[Tuple[str, float]]:
        pipeline = []
        if filters:
            pipeline.append({"$match": filters})
        pipeline.append({
            "$vectorSearch": {
                "index": self.index_name,
                "path": self.path,
                "queryVector": vector.values,
                "numCandidates": max(50, top_k * 5),
                "limit": top_k
            }
        })
        pipeline.append({"$project": {"_id": 1, "score": {"$meta": "vectorSearchScore"}}})

        # motor 的 aggregate 可直接异步迭代
        cursor = self.col.aggregate(pipeline)
        results: List[Tuple[str, float]] = []
        async for d in cursor:
            results.append((str(d["_id"]), float(d.get("score", 0.0))))
        return results

    @staticmethod
    def _to_id(doc_id: str):
        try:
            return ObjectId(doc_id)
        except Exception:
            # 如果你的 _id 本来就是字符串，这里直接返回字符串
            return doc_id