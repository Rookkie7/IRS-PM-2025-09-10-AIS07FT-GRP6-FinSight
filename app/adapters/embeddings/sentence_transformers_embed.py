import asyncio
from typing import List
# from sentence_transformers import SentenceTransformer
from typing import List, Optional
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer

class LocalEmbeddingProvider:
# class SentenceTransformerEmbedder:
    """
    sentence-transformers 真嵌入器（默认 all-MiniLM-L6-v2）。
    输出向量做 L2 归一化，便于余弦相似度。
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # 延迟导入
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_text(self, text: str) -> List[float]:
        emb = self.model.encode(text or "", normalize_embeddings=False)
        emb = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(emb) + 1e-12
        return (emb / norm).tolist()
