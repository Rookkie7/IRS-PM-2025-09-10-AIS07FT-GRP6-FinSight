from typing import List, Tuple, Optional
from app.model.models import EmbeddingVector

class VectorIndexPort:
    async def upsert(self, doc_id: str, vector: EmbeddingVector, metadata: dict) -> None: ...

    async def search(self, vector: EmbeddingVector, top_k: int = 10,
                     filters: Optional[dict] = None) -> List[Tuple[str, float]]: ...