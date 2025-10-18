# app/forecasters/lstm_forecaster.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple
from dataclasses import dataclass
import math

class LstmForecaster:
    """
    轻量 LSTM 预测器
    - 如果 __init__ 传入 self.model（nn.Module），则直接用于推理
    - 否则根据传入的 closes 做一次“快速本地训练”，然后递归多步预测
    """

    name = "lstm"

    def __init__(
        self,
        model=None,
        device: str = "cpu",
        lookback: int = 32,        # 滑窗长度
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        epochs: int = 60,          # 轻训练轮数（不要太大）
        lr: float = 1e-3,
        train_split: float = 0.85, # 训练/验证划分
        patience: int = 8,         # 早停
        min_train_points: int = 80 # 最少样本点，太少直接回退
    ):
        try:
            import torch  # noqa: F401
            import numpy as np  # noqa: F401
        except Exception as e:
            raise RuntimeError("PyTorch 或 numpy 未安装，请先 `pip install torch numpy`") from e

        self.model = model
        self.device = device
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.train_split = train_split
        self.patience = patience
        self.min_train_points = min_train_points

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

        # 小样本兜底：不足 min_train_points + lookback，直接均值回归
        if self.model is None and closes.size < max(self.min_train_points, self.lookback + 5):
            return self._mean_revert_baseline(closes, horizon)

        # 预训练模型路径（仅推理）
        if self.model is not None:
            return self._predict_with_model(self.model, closes, horizon)

        # 轻训练路径
        model, scaler = self._fit_quick(closes)
        if model is None:
            return self._mean_revert_baseline(closes, horizon)

        yhat, conf = self._roll_forecast(model, scaler, closes, horizon)
        return yhat, conf

    # ---------------------- internals ----------------------

    def _mean_revert_baseline(self, arr, horizon: int) -> Tuple[float, Optional[float]]:
        """
        占位：当前价向近 20 日均值温和回归
        """
        import numpy as np
        cur = float(arr[-1])
        ma = float(arr[-min(20, arr.size):].mean())
        alpha = min(1.0, 0.05 * max(1, horizon))
        yhat = (1 - alpha) * cur + alpha * ma
        conf = self._confidence_from_vol(arr, yhat, horizon)
        return yhat, conf

    def _prepare_supervised(self, series):
        """
        将 1D 序列 -> (X, y) 监督学习样本
        X: (N, lookback, 1) ; y: (N, 1)
        """
        import numpy as np
        lookback = self.lookback
        X, y = [], []
        for i in range(lookback, len(series)):
            X.append(series[i - lookback:i])
            y.append(series[i])
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        # shape to (N, T, 1)
        X = X[..., None]
        return X, y

    def _scale_fit(self, arr):
        """
        简易 MinMax scaler（只用训练集范围）
        """
        a_min = float(arr.min())
        a_max = float(arr.max())
        scale = max(1e-8, a_max - a_min)

        def transform(x):
            return (x - a_min) / scale

        def inverse(x):
            return x * scale + a_min

        return transform, inverse

    def _fit_quick(self, closes):
        """
        即时小模型轻训练（仅本次序列使用）
        """
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        # 训练/验证划分
        n = closes.size
        split = int(n * self.train_split)
        train_arr = closes[:max(split, self.lookback + 1)]
        val_arr = closes[max(self.lookback + 1, split):]

        # 缩放（只用训练集拟合）
        fwd, inv = self._scale_fit(train_arr)
        train_scaled = fwd(train_arr)
        Xtr, ytr = self._prepare_supervised(train_scaled)
        if Xtr.shape[0] < 10:
            return None, None

        # 构造验证集（可为空）
        Xval = yval = None
        if val_arr.size > self.lookback + 1:
            val_scaled = fwd(val_arr)
            Xval, yval = self._prepare_supervised(val_scaled)

        # 张量与加载器
        device = torch.device(self.device)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
        ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
        ds_tr = TensorDataset(Xtr_t, ytr_t)
        dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True, drop_last=False)

        if Xval is not None:
            Xval_t = torch.tensor(Xval, dtype=torch.float32, device=device)
            yval_t = torch.tensor(yval, dtype=torch.float32, device=device)

        # 小 LSTM
        class _TinyLSTM(nn.Module):
            def __init__(self, in_dim=1, hidden=self.hidden_size, layers=self.num_layers, drop=self.dropout):
                super().__init__()
                self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=layers,
                                    batch_first=True, dropout=(drop if layers > 1 else 0.0))
                self.head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x):
                out, _ = self.lstm(x)         # (B, T, H)
                last = out[:, -1, :]          # (B, H)
                y = self.head(last)           # (B, 1)
                return y

        model = _TinyLSTM().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.SmoothL1Loss()

        best_state = None
        best_val = float("inf")
        wait = 0

        # 轻训练
        for epoch in range(self.epochs):
            model.train()
            running = 0.0
            for xb, yb in dl_tr:
                optim.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optim.step()
                running += float(loss.item()) * xb.size(0)
            train_loss = running / len(ds_tr)

            # 验证
            if Xval is not None:
                model.eval()
                with torch.no_grad():
                    val_pred = model(Xval_t)
                    val_loss = float(loss_fn(val_pred, yval_t).item())
                metric = val_loss
            else:
                metric = train_loss

            # 早停
            if metric < best_val - 1e-5:
                best_val = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # 返回模型 + 反缩放函数
        class _Scaler:
            def __init__(self, fwd, inv):
                self.fwd = fwd
                self.inv = inv

        return model, _Scaler(fwd, inv)

    def _roll_forecast(self, model, scaler, closes, horizon: int) -> Tuple[float, Optional[float]]:
        """
        递归多步预测：每次预测 1 步，把预测值拼到窗口尾部再预测下一步
        """
        import numpy as np
        import torch
        device = torch.device(self.device)
        horizon = max(1, int(horizon))

        # 用完整序列最后 lookback 段做起始窗口（在 scaler 上变换）
        base = closes.astype(float)
        base = base[np.isfinite(base)]
        window = base[-self.lookback:].copy()
        fwd, inv = scaler.fwd, scaler.inv

        # 如果窗口长度不足，做 padding（用窗口均值）
        if window.size < self.lookback:
            pad_val = window.mean() if window.size > 0 else base[-1]
            window = np.pad(window, (self.lookback - window.size, 0), constant_values=pad_val)

        # 变换到 [0,1]
        win_scaled = fwd(window)
        preds_scaled = []

        model.eval()
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(win_scaled, dtype=torch.float32, device=device).view(1, -1, 1)
                y1 = model(x).view(-1).item()  # 标量（缩放域）
                preds_scaled.append(y1)

                # 更新窗口（缩放域）
                win_scaled = list(win_scaled[1:]) + [y1]
                win_scaled = np.asarray(win_scaled, dtype=float)

        # 反缩放预测序列
        preds = inv(np.asarray(preds_scaled, dtype=float))
        yhat = float(preds[-1])

        # 置信度：结合近期波动与递归序列的内部方差
        conf = self._confidence_from_series(base, preds, horizon, yhat)
        return yhat, conf

    def _predict_with_model(self, model, closes, horizon: int) -> Tuple[float, Optional[float]]:
        """
        已有预训练模型：只做递归推理（需要事先知道模型训练时的缩放方式；
        这里使用“在线自适应缩放”近似，若需严格一致可在 model 中内置 scaler）
        """
        import numpy as np
        scaler = type("Tmp", (), {})()
        fwd, inv = self._scale_fit(closes)
        scaler.fwd = fwd
        scaler.inv = inv
        return self._roll_forecast(model, scaler, np.asarray(closes, dtype=float), horizon)

    # ---------------------- confidence ----------------------

    def _confidence_from_vol(self, closes, yhat: float, horizon: int) -> float:
        """
        用近期日收益波动近似给置信度（区间越窄置信度越高）
        """
        import numpy as np, statistics as st
        arr = np.asarray(closes, dtype=float)
        tail = arr[-min(31, arr.size):]
        rets = []
        for p, c in zip(tail[:-1], tail[1:]):
            if p and p > 0:
                rets.append((c/p) - 1.0)
        sigma = st.pstdev(rets) if len(rets) > 1 else 0.02
        # 粗略把 horizon 的不确定性按 sqrt(h) 放大
        rel_width = max(1e-6, sigma * math.sqrt(max(1, horizon)))
        conf = 1.0 / (1.0 + 10 * rel_width)  # 宽 -> 小，窄 -> 大
        return float(max(0.0, min(1.0, conf)))

    def _confidence_from_series(self, closes, preds, horizon: int, yhat: float) -> float:
        """
        结合：
        1) 近期真实波动（std of returns）
        2) 递归预测序列内部方差（越平滑越自信）
        """
        import numpy as np, statistics as st
        conf_vol = self._confidence_from_vol(closes, yhat, horizon)

        preds = np.asarray(preds, dtype=float)
        if preds.size > 1:
            var_inside = float(np.var(np.diff(preds)))
            # 归一化（相对最终预测），避免量纲影响
            rel = var_inside / max(1e-6, yhat * yhat)
            conf_smooth = 1.0 / (1.0 + 1000 * rel)
        else:
            conf_smooth = 0.5

        conf = 0.6 * conf_vol + 0.4 * conf_smooth
        return float(max(0.0, min(1.0, conf)))
