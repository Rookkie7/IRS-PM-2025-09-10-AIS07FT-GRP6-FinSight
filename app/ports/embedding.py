from typing import List

class EmbeddingProviderPort:
    async def embed(self, texts: List[str], dim: int) -> List[List[float]]:
        ...