# app/forecasters/lstm_forecaster.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple
from dataclasses import dataclass
import math, random

class LstmForecaster:
    """
    轻量 LSTM 预测器（增强版）
    - __init__ 可注入预训练 torch.nn.Module；否则对传入 closes 做一次“快速小训练”
    - 训练：MinMax + （可选）log 变换，按时间块 block_len 训练，早停 + 裁剪
    - 预测：递归多步；支持对递归路径做 anchor 平滑
    """

    name = "lstm"

    def __init__(
        self,
        model=None,
        device: str = "cpu",
        lookback: int = 32,        # 滑窗长度（历史步）
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 80,          # 小训练轮次
        lr: float = 1e-3,
        train_split: float = 0.85, # 训练/验证划分（顺序切分）
        patience: int = 10,        # 早停
        min_train_points: int = 80,# 最少样本点，不足则回退
        block_len: int = 16,       # 时间块长度（类似 notebook 的 timestamp）
        log_price: bool = True,    # 在 log(price) 域拟合（更稳定）
        anchor_weight: float = 0.3,# 预测路径 anchor 平滑权重（0~1）
        seed: int = 42,            # 随机种子
    ):
        try:
            import torch  # noqa: F401
            import numpy as np  # noqa: F401
        except Exception as e:
            raise RuntimeError("PyTorch 或 numpy 未安装，请先 `pip install torch numpy`") from e

        self.model = model
        self.device = device
        self.lookback = int(lookback)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.train_split = float(train_split)
        self.patience = int(patience)
        self.min_train_points = int(min_train_points)
        self.block_len = int(block_len)
        self.log_price = bool(log_price)
        self.anchor_weight = float(anchor_weight)
        self.seed = int(seed)

    # ---------------------- public ----------------------

    def predict(self, closes: Sequence[float], horizon: int) -> Tuple[float, Optional[float]]:
        """
        输入：closes 升序价格序列；horizon 未来天数
        输出：(预测价, 置信度[0-1]或None)
        """
        import numpy as np
        closes = np.asarray(closes, dtype=float)
        closes = closes[np.isfinite(closes)]
        if closes.size < 5:
            return float(closes[-1]), None

        # 数据不足 → 均值回归兜底
        if self.model is None and closes.size < max(self.min_train_points, self.lookback + 5):
            return self._mean_revert_baseline(closes, horizon)

        # 预训练模型：仅推理
        if self.model is not None:
            return self._predict_with_model(self.model, closes, horizon)

        # 轻训练路径
        model, scaler, use_log = self._fit_quick(closes)
        if model is None:
            return self._mean_revert_baseline(closes, horizon)

        yhat, conf = self._roll_forecast(model, scaler, closes, horizon, use_log=use_log)
        return yhat, conf

    # ---------------------- internals ----------------------

    def _seed_everything(self):
        import torch, numpy as np, os
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

    def _mean_revert_baseline(self, arr, horizon: int) -> Tuple[float, Optional[float]]:
        """
        占位：当前价向近 20 日均值温和回归 + 置信度近似
        """
        import numpy as np
        cur = float(arr[-1])
        ma = float(arr[-min(20, arr.size):].mean())
        alpha = min(1.0, 0.05 * max(1, horizon))
        yhat = (1 - alpha) * cur + alpha * ma
        conf = self._confidence_from_vol(arr, yhat, horizon)
        return yhat, conf

    # —— 监督样本：seq-to-one —— #
    def _prepare_supervised(self, series):
        """
        1D 序列 -> (X, y)
        X: (N, lookback, 1) ; y: (N, 1) 预测下一时刻
        """
        import numpy as np
        L = self.lookback
        X, y = [], []
        for i in range(L, len(series)):
            X.append(series[i - L:i])
            y.append(series[i])
        X = np.array(X, dtype=float)[..., None]
        y = np.array(y, dtype=float).reshape(-1, 1)
        return X, y

    def _scale_fit(self, arr):
        """
        MinMax scaler（训练集范围）
        注意：如果使用 log 域，则先 log 再缩放
        """
        a_min = float(arr.min())
        a_max = float(arr.max())
        scale = max(1e-8, a_max - a_min)

        def transform(x):
            return (x - a_min) / scale

        def inverse(x):
            return x * scale + a_min

        return transform, inverse

    def _maybe_log(self, arr):
        """
        根据标志决定是否在 log 价格域拟合（仅正值可用）
        """
        import numpy as np
        if self.log_price and np.all(arr > 0):
            return np.log(arr), True
        return arr, False

    def _fit_quick(self, closes):
        """
        即时小模型轻训练（仅本序列使用），参考你 notebook 的：
          - MinMax 缩放（在 log 域可选）
          - 按 block_len（timestamp）分段训练
          - 早停 + grad clip
        """
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        self._seed_everything()

        # 可能的 log 域
        series, use_log = self._maybe_log(closes)

        # 训练/验证划分
        n = series.size
        split = int(n * self.train_split)
        train_arr = series[:max(split, self.lookback + 1)]
        val_arr = series[max(self.lookback + 1, split):]

        # 缩放（只用训练集拟合）
        fwd, inv = self._scale_fit(train_arr)
        train_scaled = fwd(train_arr)

        Xtr, ytr = self._prepare_supervised(train_scaled)
        if Xtr.shape[0] < 10:
            return None, None, use_log

        # 验证集（可为空）
        if val_arr.size > self.lookback + 1:
            val_scaled = fwd(val_arr)
            Xval, yval = self._prepare_supervised(val_scaled)
        else:
            Xval = yval = None

        # 张量与加载器（保序训练：用 block_len 切块模拟时间块训练）
        device = torch.device(self.device)

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
        ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)

        # 自定义“时间块”索引（不打乱，仿照 notebook 的 timestamp 循环）
        blocks = []
        step = max(1, self.block_len)
        for k in range(0, Xtr_t.size(0), step):
            end = min(k + step, Xtr_t.size(0))
            blocks.append((k, end))

        if Xval is not None:
            Xval_t = torch.tensor(Xval, dtype=torch.float32, device=device)
            yval_t = torch.tensor(yval, dtype=torch.float32, device=device)

        # 小 LSTM
        class _TinyLSTM(nn.Module):
            def __init__(self, in_dim=1, hidden=self.hidden_size, layers=self.num_layers, drop=self.dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=in_dim,
                    hidden_size=hidden,
                    num_layers=layers,
                    batch_first=True,
                    dropout=(drop if layers > 1 else 0.0),
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x, h=None):
                out, h = self.lstm(x, h)     # (B,T,H)
                last = out[:, -1, :]         # (B,H)
                y = self.head(last)          # (B,1)
                return y, h

        model = _TinyLSTM().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.SmoothL1Loss()

        best_state = None
        best_val = float("inf")
        wait = 0

        # 轻训练（按时间块循环）
        for _ in range(self.epochs):
            model.train()
            running = 0.0
            h_state = None
            for (s, e) in blocks:
                xb = Xtr_t[s:e]  # (B,T,1)
                yb = ytr_t[s:e]
                optim.zero_grad()
                pred, h_state = model(xb, h_state)
                # 截断隐藏状态的梯度（避免长依赖爆炸）
                if isinstance(h_state, tuple):
                    h_state = tuple(v.detach() for v in h_state)
                else:
                    h_state = h_state.detach() if h_state is not None else None

                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optim.step()
                running += float(loss.item()) * xb.size(0)

            train_loss = running / Xtr_t.size(0)

            # 验证
            if Xval is not None:
                model.eval()
                with torch.no_grad():
                    pred_val, _ = model(Xval_t)
                    val_loss = float(loss_fn(pred_val, yval_t).item())
                metric = val_loss
            else:
                metric = train_loss

            # 早停
            if metric < best_val - 1e-6:
                best_val = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # 返回模型 + 反缩放函数 + 是否 log 域
        class _Scaler:
            def __init__(self, fwd, inv):
                self.fwd = fwd
                self.inv = inv

        return model, _Scaler(fwd, inv), use_log

    def _anchor(self, arr, weight: float):
        """与 notebook 一致的 anchor 平滑（指数滑动）"""
        if not arr:
            return arr
        buf = []
        last = arr[0]
        for v in arr:
            sm = last * weight + (1 - weight) * v
            buf.append(sm)
            last = sm
        return buf

    def _roll_forecast(self, model, scaler, closes, horizon: int, use_log: bool) -> Tuple[float, Optional[float]]:
        """
        递归多步预测：在“scaled 域”递归，每步把预测值拼到窗口尾，再预测下一步；
        末尾把整段预测序列做 anchor 平滑（可选），最后反缩放&反 log 回价格域。
        """
        import numpy as np
        import torch

        device = torch.device(self.device)
        horizon = max(1, int(horizon))

        # 训练时的同态变换
        series = np.asarray(closes, dtype=float)
        series = series[np.isfinite(series)]
        if use_log:
            series = np.log(series)
        fwd, inv = scaler.fwd, scaler.inv

        # 初始窗口（scaled）
        window = series[-self.lookback:].copy()
        if window.size < self.lookback:
            pad = window.mean() if window.size > 0 else series[-1]
            window = np.pad(window, (self.lookback - window.size, 0), constant_values=pad)
        win_scaled = fwd(window)

        preds_scaled = []
        h_state = None
        model.eval()

        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(win_scaled, dtype=torch.float32, device=device).view(1, -1, 1)
                y1_scaled, h_state = model(x, h_state)
                y1 = float(y1_scaled.view(-1).item())
                preds_scaled.append(y1)

                # 更新窗口
                win_scaled = list(win_scaled[1:]) + [y1]
                win_scaled = np.asarray(win_scaled, dtype=float)

        # 反缩放（到 log 或线性域）
        preds = scaler.inv(np.asarray(preds_scaled, dtype=float))

        # anchor 平滑（在同域下）
        if self.anchor_weight > 0.0:
            preds = self._anchor(list(preds), weight=self.anchor_weight)

        # 若在 log 域 → exp 回价格域
        if use_log:
            preds = np.exp(np.asarray(preds, dtype=float))

        yhat = float(preds[-1])

        # 置信度：结合波动 + 递归路径内部平滑度
        conf = self._confidence_from_series(np.asarray(closes, dtype=float), np.asarray(preds, dtype=float), horizon, yhat)
        return yhat, conf

    def _predict_with_model(self, model, closes, horizon: int) -> Tuple[float, Optional[float]]:
        """
        预训练模型：采用“在线缩放”近似（严格一致性需在模型内保存 scaler）
        """
        import numpy as np
        # 判断是否可用 log 域（预训练不一定一致，这里用启发式：全正→允许 log）
        series = np.asarray(closes, dtype=float)
        use_log = self.log_price and np.all(series > 0)

        # 在线拟合 scaler（用全部历史，实际生产建议随模型保存）
        fwd, inv = self._scale_fit(np.log(series) if use_log else series)
        scaler = type("Tmp", (), {})()
        scaler.fwd, scaler.inv = fwd, inv

        return self._roll_forecast(model, scaler, series, horizon, use_log=use_log)

    # ---------------------- confidence ----------------------

    def _confidence_from_vol(self, closes, yhat: float, horizon: int) -> float:
        """
        近期日收益波动作为不确定性 proxy（按 sqrt(h) 放大）
        """
        import numpy as np, statistics as st
        arr = np.asarray(closes, dtype=float)
        tail = arr[-min(31, arr.size):]
        rets = []
        for p, c in zip(tail[:-1], tail[1:]):
            if p and p > 0:
                rets.append((c/p) - 1.0)
        sigma = st.pstdev(rets) if len(rets) > 1 else 0.02
        rel_width = max(1e-6, sigma * math.sqrt(max(1, horizon)))
        conf = 1.0 / (1.0 + 10 * rel_width)  # 宽 -> 小，窄 -> 大
        return float(max(0.0, min(1.0, conf)))

    def _confidence_from_series(self, closes, preds, horizon: int, yhat: float) -> float:
        """
        合成置信度：
        1) 真实序列的波动（std of returns）
        2) 递归预测路径的内部方差（越平滑越自信）
        """
        import numpy as np
        conf_vol = self._confidence_from_vol(closes, yhat, horizon)

        preds = np.asarray(preds, dtype=float)
        if preds.size > 1:
            var_inside = float(np.var(np.diff(preds)))
            rel = var_inside / max(1e-6, yhat * yhat)  # 归一化
            conf_smooth = 1.0 / (1.0 + 1000 * rel)
        else:
            conf_smooth = 0.5

        conf = 0.6 * conf_vol + 0.4 * conf_smooth
        return float(max(0.0, min(1.0, conf)))
