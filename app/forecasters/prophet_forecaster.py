# app/forecasters/prophet_forecaster.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple
from dataclasses import dataclass
import math
import hashlib

@dataclass
class _Cache:
    key: str | None = None
    # 轻量缓存：如果传入 closes 与上次一致，就复用同一个模型
    model: any = None
    last_y0: float | None = None  # 记录拟合时最后一个观测的“价格”（线性域）

class ProphetForecaster:  # 不直接继承，避免导入时因缺包报错
    name = "prophet"

    def __init__(self):
        try:
            import pandas as pd  # noqa: F401
            from prophet import Prophet  # noqa: F401
        except Exception as e:
            raise RuntimeError("Prophet 未安装，请先 `pip install prophet pandas`") from e

        import pandas as pd
        from prophet import Prophet
        self._pd = pd
        self._Prophet = Prophet
        self._cache = _Cache()

    def _series_key(self, closes: Sequence[float]) -> str:
        # 用长度+末端若干值构造一个轻量 key，避免计算巨大的 hash
        tail = closes[-20:] if len(closes) >= 20 else list(closes)
        payload = f"{len(closes)}|" + "|".join(f"{x:.6f}" for x in tail)
        return hashlib.md5(payload.encode()).hexdigest()

    def _build_df(self, closes: Sequence[float]) -> "self._pd.DataFrame":
        """
        使用交易日频率（Business Day 'B'）构造 ds，
        对价格做 log 变换： y = log(price)
        """
        pd = self._pd
        n = len(closes)
        end = pd.Timestamp.utcnow().normalize()
        ds = pd.bdate_range(end=end, periods=n, freq="B")
        # 保证严格升序（bdate_range 已升序，这里只是显式）
        ds = ds.sort_values()
        df = pd.DataFrame({"ds": ds, "y": closes})
        # 清理非正数，防止 log 失败
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["y"])
        df = df[df["y"] > 0]
        # log 变换
        df["y"] = df["y"].apply(lambda v: math.log(float(v)))
        return df

    def _adaptive_params(self, closes: Sequence[float]) -> dict:
        """
        根据近端波动率调参：
        - 波动大 → 更宽的 changepoint_prior_scale（允许更多趋势拐点）
        - 乘法季节性对股价更符合常识
        """
        import statistics
        if len(closes) > 3:
            tail = closes[-min(60, len(closes)-1):]
            rets = []
            for p, c in zip(tail[:-1], tail[1:]):
                if p and p > 0:
                    rets.append((c/p) - 1.0)
            vol = statistics.pstdev(rets) if len(rets) > 1 else 0.0
        else:
            vol = 0.0

        # 将日波动率映射到 cps（经验范围 0.05 ~ 0.5）
        cps = min(0.5, max(0.05, vol * 10 + 0.05))
        seasonality_mode = "multiplicative"

        return dict(
            changepoint_prior_scale=cps,
            seasonality_mode=seasonality_mode,
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
        )

    def _fit(self, df: "self._pd.DataFrame") -> "self._Prophet":
        params = self._adaptive_params(df["y"].tolist())
        m = self._Prophet(
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_mode=params["seasonality_mode"],
            weekly_seasonality=params["weekly_seasonality"],
            yearly_seasonality=params["yearly_seasonality"],
            daily_seasonality=params["daily_seasonality"],
        )
        # 需要时可添加假日、附加回归量（示例注释）
        # m.add_country_holidays(country_name="US")
        m.fit(df)
        return m

    def _conf_from_interval(self, yhat: float, lower: float, upper: float) -> float | None:
        """
        用相对区间宽度映射到 [0,1] 的直观置信度。
        区间越窄，置信度越高。保护负数/零。
        """
        try:
            width = float(upper - lower)
            denom = max(1e-8, abs(yhat))
            rel = width / denom  # 区间相对宽度
            c = 1.0 / (1.0 + rel)  # 宽->小，窄->大
            return max(0.0, min(1.0, c))
        except Exception:
            return None

    def predict(self, closes: Sequence[float], horizon: int) -> Tuple[float, Optional[float]]:
        """
        输入：closes（升序收盘价），horizon（天）
        输出： (预测价 yhat, 置信度 conf ∈ [0,1] 或 None)
        """
        # 小样本兜底
        if not closes or len(closes) < 10:
            return float(closes[-1]), None

        pd = self._pd
        horizon = max(1, int(horizon))

        # 轻量缓存：相同序列重复调用 predict 时复用模型
        key = self._series_key(closes)
        reuse = (self._cache.key == key and self._cache.model is not None)

        if reuse:
            m = self._cache.model
        else:
            df = self._build_df(closes)
            if len(df) < 10:
                return float(closes[-1]), None
            m = self._fit(df)
            self._cache.key = key
            self._cache.model = m
            # 记录拟合时最后一个“线性域价格”，用于 sanity-check
            self._cache.last_y0 = float(closes[-1])

        # 使用交易日频率扩展未来
        future = m.make_future_dataframe(periods=horizon, freq="B")
        fcst = m.predict(future)

        # 取恰好 horizon 天后的那行（尾部第 1 行）
        last = fcst.tail(1).iloc[0]
        yhat_log = float(last["yhat"])
        lower_log = float(last.get("yhat_lower", yhat_log))
        upper_log = float(last.get("yhat_upper", yhat_log))

        # 反变换回价格域
        yhat = float(math.exp(yhat_log))
        lower = float(math.exp(lower_log))
        upper = float(math.exp(upper_log))

        # 置信度
        conf = self._conf_from_interval(yhat, lower, upper)

        # 极端值保护：若预测值离当前价过远（> ±5σ 的粗略启发），做温和回缩
        try:
            cur = float(closes[-1])
            # 粗估波动：最近 30 个日收益的 std
            tail = closes[-min(31, len(closes)):]
            rets = []
            for p, c in zip(tail[:-1], tail[1:]):
                if p and p > 0:
                    rets.append((c/p) - 1.0)
            import statistics as _st
            sigma = _st.pstdev(rets) if len(rets) > 1 else 0.02
            max_dev = 5.0 * sigma
            if sigma > 0:
                dev = (yhat / cur) - 1.0
                if abs(dev) > max_dev:
                    # 回缩到边界
                    yhat = cur * (1.0 + math.copysign(max_dev, dev))
        except Exception:
            pass

        return yhat, conf
