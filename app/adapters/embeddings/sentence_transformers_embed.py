from __future__ import annotations
import numpy as np
import asyncio
from typing import List, Optional

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
# from typing import List, Optional
# import asyncio
# import numpy as np
# from sentence_transformers import SentenceTransformer

# class LocalEmbeddingProvider:
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
#         self.model_name = model_name
#         self.device = device
#         self._model = None  # 延迟加载

#     def _ensure_model(self):
#         if self._model is None:
#             # 延迟导入，减少启动时依赖链（transformers/trainer等）
#             from sentence_transformers import SentenceTransformer
#             self._model = SentenceTransformer(self.model_name, device=self.device)

#     async def embed(self, texts: List[str], dim: int = 32) -> List[List[float]]:
#         """
#         异步接口：返回每个文本的 dim 维向量
#         """
#         self._ensure_model()
#         loop = asyncio.get_event_loop()
#         # encode 是阻塞的，放线程池
#         vecs = await loop.run_in_executor(None, self._model.encode, texts, )

#         arr = np.array(vecs, dtype=np.float32)  # (n, d0)
#         d0 = arr.shape[1]
#         if d0 > dim:
#             arr = arr[:, :dim]
#         elif d0 < dim:
#             pad = np.zeros((arr.shape[0], dim - d0), dtype=np.float32)
#             arr = np.hstack([arr, pad])

#         return arr.tolist()
