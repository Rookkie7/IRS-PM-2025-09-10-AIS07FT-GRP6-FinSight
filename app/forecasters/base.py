# app/forecasters/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Optional, Tuple

class Forecaster(ABC):
    name: str  # strategy name（method parameter）

    @abstractmethod
    def predict(
        self,
        closes: Sequence[float],  # 升序的收盘价序列
        horizon: int              # 预测步数（天）
    ) -> Tuple[float, Optional[float]]:
        """返回 (预测价格, 置信度[0-1]或None)"""
        ...
