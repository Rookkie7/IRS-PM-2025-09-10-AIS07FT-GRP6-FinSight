from __future__ import annotations
from typing import List
import numpy as np
import re

class HashingEmbedder:
    """
    轻量可运行的占位嵌入器：
    - 将文本分词后用简单哈希投影到固定维度（默认32）
    - L2 归一化，便于余弦相似度
    未来可无缝替换为 sentence-transformers / bge / OpenAI 等（维度改写成一致即可）
    """
    def __init__(self, dim: int = 32, seed: int = 42):
        self.dim = dim
        self._seed = seed

    def _tokenize(self, text: str):
        return re.findall(r"[A-Za-z0-9']+", (text or "").lower())

    def embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokenize(text):
            h = hash((tok, self._seed)) % self.dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec) + 1e-12
        return (vec / norm).tolist()
