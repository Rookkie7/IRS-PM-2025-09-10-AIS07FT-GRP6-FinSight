# app/forecasters/lgbm_forecaster.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

@dataclass
class _TrainReport:
    rmse_val: float | None = None
    last_level: float | None = None

class LgbmForecaster:
    """
    轻量 LightGBM 预测器（单变量价格序列）
    - 若传入 booster（已训练模型），直接用于推理
    - 否则基于当前序列“快速训练”一个小模型，再做递归多步预测
    """
    name = "lgbm"

    def __init__(
        self,
        booster=None,
        device: str = "cpu",          # "cpu" 或 "gpu"（如安装了 GPU 版本）
        n_lags: int = 30,             # 使用多少个滞后步
        roll_windows: List[int] = [5, 10, 20],  # 滚动窗口
        train_split: float = 0.85,    # 训练/验证划分
        min_points: int = 120,        # 最少样本点；不足则回退
        num_boost_round: int = 400,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 60,
        seed: int = 42
    ):
        try:
            import lightgbm as lgb  # noqa: F401
            import numpy as np      # noqa: F401
            import pandas as pd     # noqa: F401
        except Exception as e:
            raise RuntimeError("LightGBM / numpy / pandas 未安装，请先 `pip install lightgbm numpy pandas`") from e

        self.model = booster
        self.device = device
        self.n_lags = int(n_lags)
        self.roll_windows = list(roll_windows or [])
        self.train_split = float(train_split)
        self.min_points = int(min_points)
        self.num_boost_round = int(num_boost_round)
        self.learning_rate = float(learning_rate)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.seed = int(seed)

        self._report = _TrainReport()

    # ---------------------- public ----------------------

    def predict(self, closes: Sequence[float], horizon: int) -> Tuple[float, Optional[float]]:
        """
        输入：closes 升序价格序列；horizon 未来天数
        输出：(预测价, 置信度[0-1]或 None)
        """
        import numpy as np
        arr = np.asarray(closes, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 5:
            return float(arr[-1]), None

        # 小样本兜底
        if self.model is None and arr.size < max(self.min_points, self.n_lags + 20):
            return self._baseline(arr, horizon)

        # 已有模型：滚动预测
        if self.model is not None:
            yhat = self._roll_predict(self.model, arr, horizon)
            conf = self._confidence(arr, yhat, horizon, self._report.rmse_val)
            return yhat, conf

        # 轻训练 + 滚动预测
        model = self._fit_quick(arr)
        if model is None:
            return self._baseline(arr, horizon)

        yhat = self._roll_predict(model, arr, horizon)
        conf = self._confidence(arr, yhat, horizon, self._report.rmse_val)
        return yhat, conf

    # ---------------------- internals ----------------------

    def _baseline(self, arr, horizon: int) -> Tuple[float, Optional[float]]:
        """
        占位：近 20 日均值回归 + 微弱趋势外推
        """
        import numpy as np
        cur = float(arr[-1])
        ma = float(arr[-min(20, arr.size):].mean())
        base = 0.7 * cur + 0.3 * ma
        window = min(20, arr.size)
        trend = float(arr[-1] - arr[-window]) / max(1, window - 1)
        yhat = base + trend * max(1, horizon)
        conf = self._confidence(arr, yhat, horizon, rmse_val=None)
        return yhat, conf

    def _make_feat_df(self, series) -> "pd.DataFrame":
        """
        构造监督学习特征：
        - lag1..lagN
        - 滚动均值/标准差/动量（各窗口）
        - 目标 y 为下一期价格（t）
        """
        import numpy as np
        import pandas as pd

        s = pd.Series(np.asarray(series, dtype=float), name="price")
        df = pd.DataFrame({"price": s})

        # 滞后特征
        for k in range(1, self.n_lags + 1):
            df[f"lag_{k}"] = df["price"].shift(k)

        # 滚动统计与动量
        for w in self.roll_windows:
            df[f"roll_mean_{w}"] = df["price"].rolling(w).mean()
            df[f"roll_std_{w}"]  = df["price"].rolling(w).std()
            df[f"momentum_{w}"]  = df["price"] / df["price"].shift(w) - 1.0

        # 目标：下一期价格
        df["y"] = df["price"].shift(-1)

        # 去掉前期缺失
        df = df.dropna().reset_index(drop=True)
        return df

    def _fit_quick(self, closes):
        """
        使用历史序列快速训练一个 LGBM 回归器
        """
        import numpy as np
        import pandas as pd
        import lightgbm as lgb

        df = self._make_feat_df(closes)
        if df.empty:
            return None

        # 训练/验证划分（时间序列保持顺序）
        n = len(df)
        split = max(int(n * self.train_split), 10)
        train_df = df.iloc[:split].copy()
        valid_df = df.iloc[split:].copy() if split < n else None

        X_tr = train_df.drop(columns=["y"])
        y_tr = train_df["y"].values

        if valid_df is not None and not valid_df.empty:
            X_val = valid_df.drop(columns=["y"])
            y_val = valid_df["y"].values
        else:
            X_val = None
            y_val = None

        # LightGBM 数据集
        train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
        valid_sets = [train_set]
        valid_names = ["train"]

        if X_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, free_raw_data=True)
            valid_sets.append(val_set)
            valid_names.append("valid")

        # 参数：小模型、偏稳健；device 可选 cpu/gpu
        params: Dict[str, object] = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": self.seed,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_pre_filter": False,
            "learning_rate": self.learning_rate,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "device_type": self.device,     # "cpu" / "gpu"
        }

        model = lgb.train(
            params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # 记录验证 RMSE（用于置信度）
        rmse_val = None
        try:
            if X_val is not None:
                pred_val = model.predict(X_val, num_iteration=model.best_iteration)
                rmse_val = float(((pred_val - y_val) ** 2).mean() ** 0.5)
        except Exception:
            pass

        self.model = model
        self._report = _TrainReport(rmse_val=rmse_val, last_level=float(closes[-1]))
        return model

    def _one_step_features(self, history: List[float]) -> "pd.DataFrame":
        """
        用当前历史（含预测补进去的值）构造“下一步”预测所需的一行特征
        """
        import pandas as pd
        df = self._make_feat_df(history)
        # _make_feat_df 目标 y 是 shift(-1)，因此最后一行的特征用于预测下一步
        # 取最后一行（丢 y 列）
        row = df.drop(columns=["y"]).iloc[[-1]].copy()
        return row

    def _roll_predict(self, model, closes, horizon: int) -> float:
        """
        递归多步：不断把上一步的预测值追加到历史，再预测下一步
        """
        import numpy as np
        history = list(np.asarray(closes, dtype=float))
        horizon = max(1, int(horizon))

        for _ in range(horizon):
            # 构造单行特征
            try:
                X_next = self._one_step_features(history)
            except Exception:
                # 历史太短/特征不足，退化为占位
                return self._baseline(np.asarray(history, dtype=float), horizon)[0]

            y1 = float(model.predict(X_next, num_iteration=getattr(model, "best_iteration", None))[0])
            # 保护：避免出现非正异常值（价格域通常 > 0）
            if not math.isfinite(y1) or y1 <= 0:
                y1 = max(history[-1], 1e-6)
            history.append(y1)

        return float(history[-1])

    # ---------------------- confidence ----------------------

    def _confidence(self, closes, yhat: float, horizon: int, rmse_val: float | None) -> float:
        """
        置信度合成：
        - 验证集 RMSE 相对当前价格的比例（越小越自信）
        - 近期日收益波动（越小越自信），按 sqrt(h) 放大
        """
        import numpy as np, statistics as st
        arr = np.asarray(closes, dtype=float)
        cur = float(arr[-1])

        # 基于验证 RMSE
        if rmse_val is not None and cur > 0:
            rel_err = min(1.0, rmse_val / cur)
            conf_val = 1.0 / (1.0 + 3.0 * rel_err)
        else:
            conf_val = 0.5

        # 基于波动率
        tail = arr[-min(31, arr.size):]
        rets = []
        for p, c in zip(tail[:-1], tail[1:]):
            if p and p > 0:
                rets.append((c / p) - 1.0)
        sigma = st.pstdev(rets) if len(rets) > 1 else 0.02
        rel_width = sigma * math.sqrt(max(1, horizon))
        conf_vol = 1.0 / (1.0 + 10 * rel_width)

        conf = 0.6 * conf_val + 0.4 * conf_vol
        return float(max(0.0, min(1.0, conf)))
