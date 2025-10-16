# app/forecasters/lstm_forecaster.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple

class LstmForecaster:
    name = "lstm"

    def __init__(self, model=None, device: str = "cpu"):
        try:
            import torch  # noqa: F401
        except Exception as e:
            raise RuntimeError("PyTorch 未安装，请先 pip install torch") from e
        self.model = model
        self.device = device

    def predict(self, closes: Sequence[float], horizon: int) -> Tuple[float, Optional[float]]:
        # 没有训练好的 LSTM 时，用“持有 + 轻微回归均值”的占位，保证链路通
        import numpy as np
        arr = np.asarray(closes, dtype=float)
        if arr.size < 5:
            return float(arr[-1]), None
        ma = float(arr[-min(20, arr.size):].mean())
        alpha = min(1.0, 0.05 * horizon)
        yhat = (1 - alpha) * float(arr[-1]) + alpha * ma
        return yhat, None
