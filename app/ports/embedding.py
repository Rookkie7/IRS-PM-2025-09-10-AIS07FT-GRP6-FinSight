from typing import List

class EmbeddingProviderPort:
    ...

from __future__ import annotations
from typing import Protocol, List

class EmbeddingProviderPort(Protocol):
    """嵌入提供者端口（后续可换 sentence-transformers / OpenAI embeddings）"""
    dim: int
    def embed_text(self, text: str) -> List[float]: ...
    async def embed(self, texts: List[str], dim: int) -> List[List[float]]:
        ...
