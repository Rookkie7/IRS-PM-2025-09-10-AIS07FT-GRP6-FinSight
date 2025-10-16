# app/forecasters/arima_forecaster.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import Forecaster

class ArimaForecaster(Forecaster):
    name = "arima"

    def __init__(self, order=(1,1,1)):
        self.order = order

    def predict(self, closes: Sequence[float], horizon: int) -> Tuple[float, Optional[float]]:
        y = np.asarray(closes, dtype=float)
        if y.size < 10:
            # 数据太短，退回最后一个值
            return float(y[-1]), None
        model = SARIMAX(y, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=max(1, horizon))
        mean = float(fc.predicted_mean[-1])
        # 用置信区间宽度粗略成一个“置信度”
        ci = fc.conf_int(alpha=0.2)  # 80%区间
        lower, upper = float(ci[-1,0]), float(ci[-1,1])
        width = max(1e-6, upper - lower)
        # 区间越窄置信度越高（简单映射）
        conf = max(0.0, min(1.0, 1.0 / (1.0 + width / max(1.0, abs(mean)))))
        return mean, conf
