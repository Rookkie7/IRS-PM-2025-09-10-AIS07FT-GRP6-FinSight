from __future__ import annotations
from typing import Optional
import numpy as np
from abc import ABC, abstractmethod

class BaseProjector(ABC):
    """投影器抽象：可选fit，必须transform。对外暴露dim与name，便于调试/热切换。"""
    def __init__(self, dim: int):
        self.dim = dim

    @property
    @abstractmethod
    def name(self) -> str: ...

    def fit(self, X: np.ndarray) -> None:
        """可选：PCA等需要先拟合。SRP不需要。"""
        return

    @abstractmethod
    def transform(self, v: np.ndarray) -> np.ndarray: ...


class SRPProjector(BaseProjector):
    """Sparse Random Projection (Johnson–Lindenstrauss)。无需训练、可设随机种子。"""
    def __init__(self, input_dim: int, output_dim: int = 32, seed: int = 42, density: float = 0.1):
        super().__init__(output_dim)
        assert 0 < density <= 1.0
        self._rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 三值稀疏矩阵 {-1, 0, +1} / sqrt(k)
        k = int(round(density * input_dim))
        k = max(k, 1)
        W = np.zeros((output_dim, input_dim), dtype=np.float32)
        for i in range(output_dim):
            idx = self._rng.choice(input_dim, size=k, replace=False)
            signs = self._rng.choice([-1.0, 1.0], size=k).astype(np.float32)
            W[i, idx] = signs
        self.W = W / np.sqrt(k)

    @property
    def name(self) -> str:
        return f"srp_{self.input_dim}to{self.output_dim}"

    def transform(self, v: np.ndarray) -> np.ndarray:
        # v: (input_dim,)
        return self.W @ v.astype(np.float32)


class PCAProjector(BaseProjector):
    """预留：后续若要用 PCA，可在构造时传入已拟合的投影矩阵/均值向量。"""
    def __init__(self, components: np.ndarray, mean: Optional[np.ndarray] = None):
        super().__init__(components.shape[0])
        self.components = components.astype(np.float32)  # (out_dim, in_dim)
        self.mean = mean.astype(np.float32) if mean is not None else None

    @property
    def name(self) -> str:
        return f"pca_{self.components.shape[1]}to{self.components.shape[0]}"

    def transform(self, v: np.ndarray) -> np.ndarray:
        x = v.astype(np.float32)
        if self.mean is not None:
            x = x - self.mean
        # y = C * x
        return self.components @ x
