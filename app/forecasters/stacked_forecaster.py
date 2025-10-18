# app/forecasters/stacked_forecaster.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple, Dict, List
from dataclasses import dataclass
import math
import numpy as np

# 复用你已实现的基模型（若某些不可用会自动跳过）
try:
    from app.forecasters.arima_forecaster import ArimaForecaster
except Exception:
    ArimaForecaster = None

try:
    from app.forecasters.lstm_forecaster import LstmForecaster
except Exception:
    LstmForecaster = None

try:
    from app.forecasters.lgbm_forecaster import LgbmForecaster
except Exception:
    LgbmForecaster = None

try:
    from app.forecasters.prophet_forecaster import ProphetForecaster
except Exception:
    ProphetForecaster = None


@dataclass
class _OOFReport:
    model_names: List[str]
    rmse: Dict[str, float]
    weights: Dict[str, float]


class StackedForecaster:
    """
    Stacking / Blending 预测器（ARIMA + LSTM + GBDT[LightGBM] [+ Prophet 可选]）
    - 先用历史数据做一个简易“滚动起点”的 out-of-fold(OOF) 回测：
        对每个时间 t（尾部 backtest_len 个点），
        用 [0..t] 的历史对各基模型做 1步预测（h=1），收集 OOF 预测与真实值。
    - 用 OOF 预测矩阵 X 与真实 y 训练一个“元学习器”（Ridge）；若 sklearn 不可用，
      则用“基于 RMSE 的反比例权重”做加权平均（或最小二乘闭式解）。
    - 最终预测时，对目标 horizon，先让所有基模型各自给出 h 步预测，然后用学到的权重做融合。
    - 置信度：结合（1）基模型预测的离散度（越分歧置信越低）、（2）OOF 残差的RMSE（越小置信越高）。
    """

    name = "stacked"

    def __init__(
        self,
        base_models: Optional[Dict[str, object]] = None,
        backtest_len: int = 80,      # 用于OOF权重估计的尾部长度（越大越准、越慢）
        min_train_points: int = 100, # 最少点数，不足则降级为简单平均
        use_ridge: bool = True,      # 若安装 sklearn 就用 Ridge 作为元学习器
        ridge_alpha: float = 1.0,
        allow_prophet: bool = False, # 是否纳入 Prophet（安装要求较多）
        verbose: bool = False,
    ):
        self.backtest_len = int(backtest_len)
        self.min_train_points = int(min_train_points)
        self.use_ridge = bool(use_ridge)
        self.ridge_alpha = float(ridge_alpha)
        self.verbose = bool(verbose)

        # 构建默认基模型集合
        defaults: Dict[str, object] = {}
        if ArimaForecaster:   defaults["arima"]   = ArimaForecaster(order=(1, 1, 1))
        if LstmForecaster:    defaults["lstm"]    = LstmForecaster(hidden_size=32, num_layers=1, epochs=60, patience=8)
        if LgbmForecaster:    defaults["lgbm"]    = LgbmForecaster()
        if allow_prophet and ProphetForecaster:
            defaults["prophet"] = ProphetForecaster()

        # 合并外部传入
        self.base_models: Dict[str, object] = {**defaults, **(base_models or {})}

        # 存放 OOF 评估与融合权重
        self._oof: Optional[_OOFReport] = None

        # sklearn 可能不可用
        try:
            from sklearn.linear_model import Ridge  # noqa: F401
            self._sklearn_ok = True
        except Exception:
            self._sklearn_ok = False

    # ------------------ 公共接口 ------------------

    def predict(self, closes: Sequence[float], horizon: int) -> Tuple[float, Optional[float]]:
        arr = np.asarray(closes, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 5:
            return float(arr[-1]), None

        # OOF 权重估计（懒加载 / 缓存可自行扩展）
        if (self._oof is None) and (arr.size >= self.min_train_points):
            self._oof = self._fit_oof(arr)

        # 让每个基模型预测 h 步
        preds, confs = self._predict_base_bundle(arr, horizon)

        # 融合
        yhat = self._blend(preds)
        conf = self._confidence(arr, preds, confs, yhat, horizon)
        return yhat, conf

    # ------------------ OOF 权重估计 ------------------

    def _fit_oof(self, series: np.ndarray) -> Optional[_OOFReport]:
        """
        用尾部 backtest_len 做简单滚动起点的 OOF：
        对 t = n - backtest_len - 1 ... n-2:
          - 用 series[:t+1] 作为历史
          - 各基模型做 horizon=1 的预测
        """
        n = series.size
        btl = min(self.backtest_len, max(10, n // 3))  # 控制上限与下限
        start = max(20, n - btl - 1)

        model_names = [k for k in self.base_models.keys()]
        X_list: List[List[float]] = []
        y_list: List[float] = []

        # 为了效率，可缓存基模型对象（某些模型拟合代价大）
        for t in range(start, n - 1):
            hist = series[: (t + 1)]
            y_true_next = float(series[t + 1])

            row = []
            for name in model_names:
                try:
                    pred, _ = self.base_models[name].predict(hist, horizon=1)
                except Exception:
                    pred = np.nan
                row.append(float(pred) if np.isfinite(pred) else np.nan)

            if not any(np.isfinite(row)):
                continue
            X_list.append(row)
            y_list.append(y_true_next)

        if len(X_list) < 10:
            # OOF 样本太少，不做学习，返回 None 触发兜底平均
            return None

        X = np.asarray(X_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
        # 去掉含 NaN 的行
        mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]
        if X.shape[0] < 10:
            return None

        # 计算各基模型的 RMSE
        rmse: Dict[str, float] = {}
        for j, name in enumerate(model_names):
            e = y - X[:, j]
            rmse[name] = float(np.sqrt(np.mean(e * e)))

        # 元学习器：优先 Ridge（如果 sklearn 可用），否则反比例加权
        weights: Dict[str, float]
        if self._sklearn_ok and self.use_ridge:
            from sklearn.linear_model import Ridge
            # 简单标准化，避免尺度问题
            Xn = (X - X.mean(0)) / np.maximum(1e-8, X.std(0))
            model = Ridge(alpha=self.ridge_alpha, fit_intercept=True, positive=False, random_state=42)
            model.fit(Xn, y)
            # 把系数映射回“原尺度贡献”（仅用于融合；可选归一化为正权重）
            coefs = model.coef_
            # 将权重归一化（和为1；允许负的会带来振荡，这里投影到非负）
            w = np.maximum(0.0, coefs)
            if w.sum() <= 1e-12:
                # 若全非正，用 RMSE 反比例
                weights = self._rmse_inverse_weights(rmse)
            else:
                w = w / w.sum()
                weights = {name: float(w[j]) for j, name in enumerate(model_names)}
        else:
            weights = self._rmse_inverse_weights(rmse)

        if self.verbose:
            print(f"[stacked] OOF samples={X.shape[0]} rmse={rmse} weights={weights}")

        return _OOFReport(model_names=model_names, rmse=rmse, weights=weights)

    @staticmethod
    def _rmse_inverse_weights(rmse: Dict[str, float]) -> Dict[str, float]:
        # w ∝ 1 / (rmse + eps)
        eps = 1e-8
        inv = {k: 1.0 / max(eps, v) for k, v in rmse.items() if np.isfinite(v) and v > 0}
        s = sum(inv.values())
        if s <= 0:
            # 均匀权重
            m = len(rmse)
            return {k: 1.0 / m for k in rmse.keys()}
        return {k: v / s for k, v in inv.items()}

    # ------------------ 预测并融合 ------------------

    def _predict_base_bundle(self, series: np.ndarray, horizon: int) -> Tuple[Dict[str, float], Dict[str, float | None]]:
        preds: Dict[str, float] = {}
        confs: Dict[str, float | None] = {}
        for name, model in self.base_models.items():
            try:
                yhat, conf = model.predict(series, horizon=horizon)
                if not math.isfinite(float(yhat)):
                    continue
                preds[name] = float(yhat)
                confs[name] = float(conf) if (conf is not None and math.isfinite(conf)) else None
            except Exception:
                continue

        # 如果一个都没有成功，退化为持有
        if not preds:
            preds["naive"] = float(series[-1])
            confs["naive"] = None
        return preds, confs

    def _blend(self, preds: Dict[str, float]) -> float:
        # 如果没做 OOF 或基模型数量变化了，用均匀/简单权重
        if (self._oof is None) or (set(preds.keys()) != set(self._oof.model_names)):
            w = {k: 1.0 / len(preds) for k in preds.keys()}
        else:
            # 用 OOF 学到的权重
            w = {k: self._oof.weights.get(k, 0.0) for k in preds.keys()}
            # 若全为0，回退均匀
            s = sum(w.values())
            if s <= 1e-12:
                w = {k: 1.0 / len(preds) for k in preds.keys()}
            else:
                w = {k: v / s for k, v in w.items()}

        # 线性融合
        yhat = sum(w[k] * preds[k] for k in preds.keys())
        return float(yhat)

    # ------------------ 置信度估计 ------------------

    def _confidence(self, closes: np.ndarray, preds: Dict[str, float], confs: Dict[str, Optional[float]],
                    yhat: float, horizon: int) -> float:
        """
        合成置信度：
        - 基模型间分歧：std(preds) / |yhat| 越小 → 越自信
        - 若有 OOF：RMSE 的反比例作为上限
        - 近期波动：sigma*sqrt(h) 越小 → 越自信
        """
        # 分歧项
        vals = np.array([v for v in preds.values() if np.isfinite(v)], dtype=float)
        if vals.size >= 2:
            spread = float(np.std(vals))
            denom = max(1e-6, abs(yhat))
            c_spread = 1.0 / (1.0 + 5.0 * (spread / denom))  # 经验缩放
        else:
            c_spread = 0.6

        # OOF 残差项
        if self._oof is not None and len(self._oof.rmse) > 0 and abs(yhat) > 0:
            # 用权重对 rmse 做加权平均
            if self._oof.weights:
                w = np.array([self._oof.weights.get(k, 0.0) for k in self._oof.model_names], dtype=float)
                r = np.array([self._oof.rmse.get(k, np.nan) for k in self._oof.model_names], dtype=float)
                mask = np.isfinite(r) & (w > 0)
                if mask.any():
                    rmse_w = float(np.sum(w[mask] * r[mask]) / np.sum(w[mask]))
                else:
                    rmse_w = float(np.nanmean(r))
            else:
                rmse_w = float(np.nanmean(list(self._oof.rmse.values())))
            rel = min(1.0, rmse_w / max(1e-6, abs(yhat)))
            c_oof = 1.0 / (1.0 + 3.0 * rel)
        else:
            c_oof = 0.5

        # 波动项
        tail = closes[-min(31, closes.size):]
        rets = []
        for p, c in zip(tail[:-1], tail[1:]):
            if p and p > 0:
                rets.append((c / p) - 1.0)
        sigma = float(np.std(rets)) if len(rets) > 1 else 0.02
        relw = sigma * math.sqrt(max(1, horizon))
        c_vol = 1.0 / (1.0 + 10.0 * relw)

        # 合成
        conf = 0.45 * c_spread + 0.35 * c_oof + 0.20 * c_vol
        return float(max(0.0, min(1.0, conf)))
