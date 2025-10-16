# app/forecasters/lgbm_forecaster.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple

class LgbmForecaster:
    name = "lgbm"

    def __init__(self, booster=None):
        try:
            import lightgbm as lgb  # noqa: F401
        except Exception as e:
            raise RuntimeError("LightGBM 未安装，请先 pip install lightgbm") from e
        self.model = booster  # 允许外部注入已训练模型；此处演示简单规则

    def predict(self, closes: Sequence[float], horizon: int) -> Tuple[float, Optional[float]]:
        # 没有训练好的模型时，用一个非常朴素的“近 20 日均价 + 线性外推”占位
        import numpy as np
        arr = np.asarray(closes, dtype=float)
        if arr.size < 5:
            return float(arr[-1]), None
        window = min(20, arr.size)
        base = float(arr[-1])
        trend = float(arr[-1] - arr[-window]) / max(1, window-1)
        yhat = base + trend * horizon
        return yhat, None
