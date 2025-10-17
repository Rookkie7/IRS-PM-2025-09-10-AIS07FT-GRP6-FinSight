from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, Optional
import math, statistics

from app.model.models import ForecastResult, ForecastPoint
from app.forecasters.base import Forecaster
try:
    from app.forecasters.arima_forecaster import ArimaForecaster
except Exception:
    ArimaForecaster = None
try:
    from app.forecasters.prophet_forecaster import ProphetForecaster
except Exception:
    ProphetForecaster = None
try:
    from app.forecasters.lgbm_forecaster import LgbmForecaster
except Exception:
    LgbmForecaster = None
try:
    from app.forecasters.lstm_forecaster import LstmForecaster
except Exception:
    LstmForecaster = None

class PriceProviderPort:
    async def get_recent_closes(self, ticker: str, lookback_days: int = 252) -> List[float]:
        raise NotImplementedError

@dataclass
class ForecastConfig:
    lookback_days: int = 252
    ma_window: int = 20

class ForecastService:
    def __init__(
        self,
        price_provider: Optional[PriceProviderPort] = None,
        cfg: Optional[ForecastConfig] = None,
        forecasters: Optional[dict[str, Forecaster]] = None
    ):
        self.price_provider = price_provider
        self.cfg = cfg or ForecastConfig()

        default_registry: dict[str, Forecaster] = {}
        if ArimaForecaster:   default_registry["arima"]   = ArimaForecaster(order=(1,1,1))
        if ProphetForecaster: default_registry["prophet"] = ProphetForecaster()
        if LgbmForecaster:    default_registry["lgbm"]    = LgbmForecaster(booster=None)
        if LstmForecaster:    default_registry["lstm"]    = LstmForecaster(model=None)
        self.forecasters = {**default_registry, **(forecasters or {})}

    async def forecast(
        self,
        ticker: str,
        horizon_days: int = 7,
        horizons: Optional[Sequence[int]] = None,
        method: str = "naive-drift",
    ) -> ForecastResult:
        # 1) 历史价格（升序）
        if self.price_provider is not None:
            closes = await self.price_provider.get_recent_closes(
                ticker, lookback_days=self.cfg.lookback_days
            )
        else:
            closes = [100 + 0.12 * i for i in range(260)]
        if not closes or len(closes) < 3:
            raise ValueError(f"No sufficient price data for {ticker},len of closes: {len(closes)}")

        current = float(closes[-1])
        # 默认 horizons：7,30,90,180
        hs = list(horizons) if horizons else [7, 30, 90, 180]

        # 2) 基础统计（尾部窗口）
        w = max(2, min(self.cfg.ma_window, len(closes) - 1))
        rets: List[float] = []
        tail = closes[-(w+1):]
        for prev, now in zip(tail[:-1], tail[1:]):
            if prev and prev > 0:
                rets.append((now / prev) - 1.0)
        avg_daily = statistics.fmean(rets) if rets else 0.0
        vol_daily = statistics.pstdev(rets) if len(rets) > 1 else 0.0

        preds: List[ForecastPoint] = []
        for h in hs:
            if method in self.forecasters:
                yhat, conf = self.forecasters[method].predict(closes, h)
            else:
                yhat, conf = self._predict_by_method(
                    method=method, current=current, closes=closes,
                    avg_daily=avg_daily, vol_daily=vol_daily,
                    horizon=h, ma_window=self.cfg.ma_window,
                )
            preds.append(
                ForecastPoint(
                    horizon_days=h,
                    predicted=round(float(yhat), 4),
                    confidence=round(float(conf), 4) if conf is not None else None,
                )
            )

        return ForecastResult(
            ticker=ticker.upper(),
            method=method,
            generated_at=datetime.utcnow(),
            current_price=round(current, 4),
            predictions=sorted(preds, key=lambda p: p.horizon_days),
        )

    @staticmethod
    def _predict_by_method(
        method: str,
        current: float,
        closes: Sequence[float],
        avg_daily: float,
        vol_daily: float,
        horizon: int,
        ma_window: int,
    ) -> tuple[float, Optional[float]]:
        method = (method or "naive-drift").lower()

        if method == "ma":
            ma_w = min(ma_window, len(closes))
            ma = sum(closes[-ma_w:]) / ma_w if ma_w > 0 else current
            alpha = min(1.0, 0.1 + 0.02 * max(1, horizon))  # 回归强度
            yhat = (1 - alpha) * current + alpha * ma
            conf = max(0.0, min(1.0, 1.0 - min(1.0, vol_daily * math.sqrt(max(1, horizon)) * 10)))
            return yhat, conf

        # 默认：naive-drift（复合增长）
        drift = (1.0 + avg_daily) ** max(1, horizon)
        yhat = current * drift
        conf = max(0.0, min(1.0, 1.0 - min(1.0, vol_daily * math.sqrt(max(1, horizon)) * 10)))
        return yhat, conf
