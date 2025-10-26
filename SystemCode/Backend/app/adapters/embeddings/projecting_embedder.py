from __future__ import annotations
from typing import List
import numpy as np
from .projectors import SRPProjector, BaseProjector

class ProjectingEmbedder:
    """
    组合模式：先用“底层嵌入器”得到高维(如384)，再经投影(SRP/PCA)降到32并L2归一化。
    对外暴露 dim=projection_dim，保证系统内所有向量一致。
    """
    def __init__(self, base_embedder, method: str = "srp", proj_dim: int = 32, seed: int = 42):
        self.base = base_embedder
        base_dim = int(getattr(base_embedder, "dim", 0) or 0)
        if base_dim <= 0:
            raise ValueError("Base embedder must expose a positive .dim")
        method = (method or "srp").lower().strip()
        if method == "srp":
            self.projector: BaseProjector = SRPProjector(input_dim=base_dim, output_dim=proj_dim, seed=seed)
        else:
            # 预留其它方法：pca/none...
            raise ValueError(f"Unknown projection method: {method}")
        self.dim = proj_dim
        self.method = method
        self.seed = seed

    def embed_text(self, text: str) -> List[float]:
        hv = np.asarray(self.base.embed_text(text), dtype=np.float32)  # 高维，如384
        lv = self.projector.transform(hv)                               # 降到32
        # 归一化
        n = float(np.linalg.norm(lv) + 1e-12)
        return (lv / n).tolist()
