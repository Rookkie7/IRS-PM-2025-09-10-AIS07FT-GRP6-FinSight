from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, Optional, Tuple, Dict
import math, statistics, re

from app.model.models import ForecastResult, ForecastPoint
from app.forecasters.base import Forecaster

# ===== 原有可选模型 =====
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

from app.forecasters.stacked_forecaster import StackedForecaster

# ===== NEW: 引入三种深度学习预测器（之前我们给你的实现）=====
try:
    from app.forecasters.seq2seq_forecaster import Seq2SeqForecaster
except Exception:
    Seq2SeqForecaster = None
try:
    from app.forecasters.dilated_cnn_forecaster import DilatedCNNForecaster
except Exception:
    DilatedCNNForecaster = None
try:
    from app.forecasters.transformer_forecaster import TransformerForecaster
except Exception:
    TransformerForecaster = None


class PriceProviderPort:
    async def get_recent_closes(self, ticker: str, lookback_days: int = 252) -> List[float]:
        raise NotImplementedError


@dataclass
class ForecastConfig:
    lookback_days: int = 252
    ma_window: int = 20
    auto_min_len_ml: int = 120    # 长度不足时不选 ML（lgbm/lstm/深度模型）
    auto_min_len_prophet: int = 80
    auto_min_len_arima: int = 30


# ===== NEW: 轻量适配器，把我们写的 PyTorch 预测器包装为 Forecaster 接口 =====
class _DLAdapter(Forecaster):
    """
    适配我们提供的 PyTorch 预测器类（有 fit/predict 且一次性输出多步序列）。
    predict() 返回 (最后一步 yhat, conf)，其中 conf 先返回 None（由上层基于波动合成）。
    """
    def __init__(self, ctor):
        """
        ctor: 一个无参的可调用，返回某个我们实现的 Forecaster 实例
              例如 lambda: Seq2SeqForecaster(device="cpu", lookback=64, epochs=40)
        """
        self._ctor = ctor

    def predict(self, closes: Sequence[float], horizon_days: int) -> Tuple[float, Optional[float]]:
        model = self._ctor()
        # 训练用全序列，预测用末尾 lookback 段
        model.fit(closes, horizon=horizon_days)
        preds = model.predict(closes[-model.cfg.lookback:], horizon=horizon_days)
        # 取多步预测的最后一个点作为 yhat
        yhat = float(preds[-1]) if preds else float(closes[-1])
        return yhat, None  # 置信度交由上层统一计算/合成


class ForecastService:
    def __init__(
        self,
        price_provider: Optional[PriceProviderPort] = None,
        cfg: Optional[ForecastConfig] = None,
        forecasters: Optional[Dict[str, Forecaster]] = None
    ):
        self.price_provider = price_provider
        self.cfg = cfg or ForecastConfig()

        default_registry: Dict[str, Forecaster] = {}

        if ArimaForecaster:
            default_registry["arima"] = ArimaForecaster(order=(1, 1, 1))
        if ProphetForecaster:
            default_registry["prophet"] = ProphetForecaster()
        if LgbmForecaster:
            default_registry["lgbm"] = LgbmForecaster(booster=None)
        if LstmForecaster:
            default_registry["lstm"] = LstmForecaster(model=None)

        # ===== NEW: 注册三种新深度学习模型（用适配器包裹）=====
        if Seq2SeqForecaster:
            default_registry["seq2seq"] = _DLAdapter(
                lambda: Seq2SeqForecaster(device="cpu", lookback=64, epochs=40, batch_size=64)
            )
        if DilatedCNNForecaster:
            default_registry["dilated_cnn"] = _DLAdapter(
                lambda: DilatedCNNForecaster(device="cpu", lookback=64, epochs=30, batch_size=128)
            )
        if TransformerForecaster:
            default_registry["transformer"] = _DLAdapter(
                lambda: TransformerForecaster(device="cpu", lookback=96, epochs=40, batch_size=64)
            )

        # 新增堆叠（保持你原配置）
        default_registry["stacked"] = StackedForecaster(
            backtest_len=80,
            min_train_points=120,
            allow_prophet=False,  # 安装好了 Prophet 再开
            verbose=False
        )

        # 合并外部传入的 forecasters（可覆盖默认）
        self.forecasters = {**default_registry, **(forecasters or {})}

    # --------------------- public API ---------------------

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

        # 3) 解析方法：单模型 / auto / ensemble
        method_norm = (method or "naive-drift").strip().lower()
        model_list = self._resolve_methods(method_norm, series_len=len(closes), vol_daily=vol_daily)

        preds: List[ForecastPoint] = []
        for h in hs:
            yhat, conf = self._predict_multi(
                model_list=model_list,
                closes=closes,
                current=current,
                avg_daily=avg_daily,
                vol_daily=vol_daily,
                horizon=h,
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
            method=method_norm,
            generated_at=datetime.utcnow(),
            current_price=round(current, 4),
            predictions=sorted(preds, key=lambda p: p.horizon_days),
        )

    # --------------------- helpers ---------------------

    def _resolve_methods(self, method: str, series_len: int, vol_daily: float) -> List[str]:
        """
        返回用于预测的模型列表：
        - 'arima' / 'prophet' / 'lgbm' / 'lstm' / 'seq2seq' / 'dilated_cnn' / 'transformer' / 'ma' / 'naive-drift'
        - 'auto'：根据长度与波动选择
        - 'ensemble(a,b,c)'：集合多个模型（存在即用，不存在跳过；全不存在则回退 'naive-drift'）
        """
        # ensemble(x,y,z)
        m = re.match(r"^ensemble\((.*?)\)$", method)
        if m:
            items = [x.strip() for x in m.group(1).split(",") if x.strip()]
            usable = [x for x in items if (x in self.forecasters or x in ("ma", "naive-drift"))]
            return usable or ["naive-drift"]

        # auto：规则可按需扩展
        if method == "auto":
            candidates: List[str] = []
            if series_len >= self.cfg.auto_min_len_arima and "arima" in self.forecasters:
                candidates.append("arima")
            if series_len >= self.cfg.auto_min_len_prophet and "prophet" in self.forecasters:
                candidates.append("prophet")
            if series_len >= self.cfg.auto_min_len_ml and "lgbm" in self.forecasters:
                candidates.append("lgbm")
            if series_len >= self.cfg.auto_min_len_ml and "lstm" in self.forecasters:
                candidates.append("lstm")
            # ===== UPDATED: 把新深度模型纳入 auto 选择 =====
            if series_len >= self.cfg.auto_min_len_ml and "seq2seq" in self.forecasters:
                candidates.append("seq2seq")
            if series_len >= self.cfg.auto_min_len_ml and "dilated_cnn" in self.forecasters:
                candidates.append("dilated_cnn")
            if series_len >= self.cfg.auto_min_len_ml and "transformer" in self.forecasters:
                candidates.append("transformer")

            # 波动太高时，附加一个均值回归的 'ma' 作为保险
            if vol_daily > 0.03:
                candidates.append("ma")
            return candidates or ["naive-drift"]

        # 单模型或内置方法
        if method in self.forecasters or method in ("ma", "naive-drift"):
            return [method]

        # 未知方法，回退
        return ["naive-drift"]

    def _predict_multi(
        self,
        model_list: List[str],
        closes: Sequence[float],
        current: float,
        avg_daily: float,
        vol_daily: float,
        horizon: int,
    ) -> Tuple[float, Optional[float]]:
        """
        对 model_list 中的每个方法跑一遍：
        - 若为注册的 Forecaster：调用其 predict
        - 若为内置：走 _predict_by_method
        - 聚合：简单平均（你也可按历史误差做加权）
        置信度：取各模型 conf 的平均；若都没有，用基于波动的启发式。
        """
        yhats: List[float] = []
        confs: List[Optional[float]] = []

        for m in model_list:
            try:
                if m in self.forecasters:
                    yhat, conf = self.forecasters[m].predict(closes, horizon)
                else:
                    yhat, conf = self._predict_by_method(
                        method=m,
                        current=current,
                        closes=closes,
                        avg_daily=avg_daily,
                        vol_daily=vol_daily,
                        horizon=horizon,
                        ma_window=self.cfg.ma_window,
                    )
                if yhat is not None and math.isfinite(yhat):
                    yhats.append(float(yhat))
                    confs.append(conf)
            except Exception:
                # 某个模型失败时忽略它，继续用其它模型
                continue

        if not yhats:
            # 全部失败，兜底 naive-drift
            yhat, conf = self._predict_by_method(
                method="naive-drift",
                current=current,
                closes=closes,
                avg_daily=avg_daily,
                vol_daily=vol_daily,
                horizon=horizon,
                ma_window=self.cfg.ma_window,
            )
            return yhat, conf

        y = float(sum(yhats) / len(yhats))

        # 置信度合成：有值取均值；否则用基于波动的启发式
        valid_confs = [c for c in confs if c is not None and math.isfinite(c)]
        if valid_confs:
            c = float(sum(valid_confs) / len(valid_confs))
        else:
            c = max(0.0, min(1.0, 1.0 - min(1.0, vol_daily * math.sqrt(max(1, horizon)) * 10)))

        return y, c

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
