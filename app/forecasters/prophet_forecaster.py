# app/forecasters/prophet_forecaster.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple

class ProphetForecaster:  # 不直接继承，避免导入时因缺包报错
    name = "prophet"

    def __init__(self):
        try:
            import pandas as pd
            from prophet import Prophet
        except Exception as e:
            raise RuntimeError("Prophet 未安装，请先 pip install prophet pandas") from e
        self._pd = __import__("pandas")
        self._Prophet = Prophet

    def predict(self, closes: Sequence[float], horizon: int) -> Tuple[float, Optional[float]]:
        import pandas as pd
        from datetime import date, timedelta
        if len(closes) < 10:
            return float(closes[-1]), None
        start = date.today() - timedelta(days=len(closes)-1)
        df = pd.DataFrame({"ds": [start + timedelta(days=i) for i in range(len(closes))], "y": list(closes)})
        m = self._Prophet(daily_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=max(1, horizon), freq="D")
        fc = m.predict(future).tail(1).iloc[0]
        yhat = float(fc["yhat"])
        width = float(fc["yhat_upper"] - fc["yhat_lower"])
        conf = max(0.0, min(1.0, 1.0/(1.0 + width / max(1.0, abs(yhat)))))
        return yhat, conf
