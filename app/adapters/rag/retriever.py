from __future__ import annotations

from typing import List, Tuple
from app.model.models import EmbeddingVector
from app.ports.vector_index import VectorIndexPort

class Retriever:
    def __init__(self, index: VectorIndexPort, dim: int = 32):
        self.index = index
        self.dim = dim

    async def retrieve(self, query_vec: list[float], top_k: int = 5, filters: dict | None = None) -> List[Tuple[str, float]]:
        q = EmbeddingVector(dim=self.dim, values=query_vec)
        return await self.index.search(q, top_k=top_k, filters=filters or {"type": "news"})