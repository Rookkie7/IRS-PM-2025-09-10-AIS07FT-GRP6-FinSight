# app/forecasters/lstm_forecaster.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple
from dataclasses import dataclass
import math, random
import torch  # noqa: F401
import numpy as np  # noqa: F401
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
        lookback: int = 64,        # 滑窗长度（历史步）
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 100,          # 小训练轮次
        lr: float = 1e-3,
        train_split: float = 0.85, # 训练/验证划分（顺序切分）
        patience: int = 10,        # 早停
        min_train_points: int = 80,# 最少样本点，不足则回退
        block_len: int = 16,       # 时间块长度（类似 notebook 的 timestamp）
        log_price: bool = True,    # 在 log(price) 域拟合（更稳定）
        anchor_weight: float = 0.3,# 预测路径 anchor 平滑权重（0~1）
        seed: int = 42,            # 随机种子
        model_cache_dir: str = "./models/lstm",   # ← NEW
        cache_key: Optional[str] = None,          # ← NEW (比如 'AAPL')
        auto_load: bool = True,                   # ← NEW
        auto_save: bool = True,                   # ← NEW
    ):

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

        self.model_cache_dir = model_cache_dir
        self.cache_key = cache_key
        self.auto_load = bool(auto_load)
        self.auto_save = bool(auto_save)

    # ---------------------- public ----------------------

    def predict(self, closes: Sequence[float], horizon: int) -> Tuple[float, Optional[float]]:
        import numpy as np
        closes = np.asarray(closes, dtype=float)
        closes = closes[np.isfinite(closes)]
        if closes.size < 5:
            return float(closes[-1]), None

        # 优先使用缓存的最佳模型（如果配置开启且 cache_key 存在）
        if self.auto_load and self.cache_key and self.model is None:
            loaded = self._try_load_best()
            if loaded is not None:
                model, scaler, use_log = loaded
                return self._roll_forecast(model, scaler, closes, horizon, use_log=use_log)

        # 数据不足 → 均值回归兜底
        if self.model is None and closes.size < max(self.min_train_points, self.lookback + 5):
            return self._mean_revert_baseline(closes, horizon)

        # 预训练模型：仅推理
        if self.model is not None:
            return self._predict_with_model(self.model, closes, horizon)

        # 轻训练路径（会在内部维护 best_state）
        model, scaler, use_log, best_metric = self._fit_quick(closes, return_metric=True)  # ← 改动点
        if model is None:
            return self._mean_revert_baseline(closes, horizon)

        # 训练完成后再保险保存一次最佳（避免最后一次未触发 on-improve 保存）
        if self.auto_save and self.cache_key:
            try:
                self._save_best(model, scaler, use_log, best_metric)
            except Exception as _:
                pass

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

    # ===== 修改 _fit_quick：在最优时即时落盘；并返回 best_metric =====
    def _fit_quick(self, closes, return_metric: bool = False):
        import numpy as np
        import torch
        import torch.nn as nn

        self._seed_everything()
        series, use_log = self._maybe_log(closes)

        # 划分
        n = series.size
        split = int(n * self.train_split)
        train_arr = series[:max(split, self.lookback + 1)]
        val_arr = series[max(self.lookback + 1, split):]

        # 缩放（只用训练集拟合）
        fwd, inv = self._scale_fit(train_arr)
        train_scaled = fwd(train_arr)
        Xtr, ytr = self._prepare_supervised(train_scaled)
        if Xtr.shape[0] < 10:
            return (None, None, use_log, float("inf")) if return_metric else (None, None, use_log)

        if val_arr.size > self.lookback + 1:
            val_scaled = fwd(val_arr)
            Xval, yval = self._prepare_supervised(val_scaled)
        else:
            Xval = yval = None

        device = torch.device(self.device)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
        ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
        if Xval is not None:
            Xval_t = torch.tensor(Xval, dtype=torch.float32, device=device)
            yval_t = torch.tensor(yval, dtype=torch.float32, device=device)

        class _TinyLSTM(nn.Module):
            def __init__(self, in_dim=1, hidden=self.hidden_size, layers=self.num_layers, drop=self.dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=in_dim, hidden_size=hidden, num_layers=layers, batch_first=True,
                    dropout=(drop if layers > 1 else 0.0),
                )
                self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

            def forward(self, x, h=None):
                out, h = self.lstm(x, h)
                last = out[:, -1, :]
                y = self.head(last)
                return y, h

        model = _TinyLSTM().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.SmoothL1Loss()

        best_state = None
        best_val = float("inf")
        wait = 0

        # 时间块
        blocks = []
        step = max(1, self.block_len)
        for k in range(0, Xtr_t.size(0), step):
            end = min(k + step, Xtr_t.size(0))
            blocks.append((k, end))

        for _ in range(self.epochs):
            model.train()
            running = 0.0
            h_state = None
            for (s, e) in blocks:
                xb = Xtr_t[s:e]
                yb = ytr_t[s:e]
                optim.zero_grad()
                pred, h_state = model(xb, h_state)
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

            if Xval is not None:
                model.eval()
                with torch.no_grad():
                    pred_val, _ = model(Xval_t)
                    val_loss = float(loss_fn(pred_val, yval_t).item())
                metric = val_loss
            else:
                metric = train_loss

            # 早停 + 保存最佳
            if metric < best_val - 1e-6:
                best_val = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0

                # ←← NEW: 一旦提升，就立刻保存为最佳（需要 cache_key + auto_save）
                if self.auto_save and self.cache_key:
                    class _Scaler:
                        def __init__(self, fwd, inv):
                            self.fwd = fwd;
                            self.inv = inv

                    try:
                        self._save_best(model, _Scaler(fwd, inv), use_log, best_val)
                    except Exception:
                        pass

            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        class _Scaler:
            def __init__(self, fwd, inv):
                self.fwd = fwd;
                self.inv = inv

        return (model, _Scaler(fwd, inv), use_log, best_val) if return_metric else (model, _Scaler(fwd, inv), use_log)

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

    # ===== 新增：保存/加载工具函数（放在类内部任意位置，建议放到 internals 区域）=====
    def _cache_dir(self) -> Optional[str]:
        if not self.cache_key:
            return None
        import os
        d = os.path.join(self.model_cache_dir, str(self.cache_key).upper())
        os.makedirs(d, exist_ok=True)
        return d

    def _save_best(self, model, scaler, use_log: bool, best_metric: float):
        """
        保存最佳：state_dict + meta.json + scaler.npz
        """
        import os, json, torch, numpy as np, time
        d = self._cache_dir()
        if not d:
            return
        # 1) 权重
        torch.save(model.state_dict(), os.path.join(d, "model.pt"))
        # 2) scaler（从 inv 推回 a_min/scale）
        a_min = float(scaler.inv(0.0))
        a_max = float(scaler.inv(1.0))
        scale = float(max(1e-8, a_max - a_min))
        np.savez(os.path.join(d, "scaler.npz"), a_min=a_min, scale=scale)
        # 3) meta
        meta = dict(
            name=self.name,
            device=self.device,
            lookback=self.lookback,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            epochs=self.epochs,
            lr=self.lr,
            train_split=self.train_split,
            patience=self.patience,
            min_train_points=self.min_train_points,
            block_len=self.block_len,
            log_price=self.log_price,
            anchor_weight=self.anchor_weight,
            seed=self.seed,
            use_log=bool(use_log),
            best_metric=float(best_metric),
            saved_at=int(time.time()),
        )
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def _try_load_best(self):
        """
        尝试从缓存目录加载：返回 (model, scaler, use_log) 或 None
        """
        import os, json, torch, numpy as np, torch.nn as nn

        d = self._cache_dir()
        if not d:
            return None
        pt = os.path.join(d, "model.pt")
        meta_f = os.path.join(d, "meta.json")
        sc_f = os.path.join(d, "scaler.npz")
        if not (os.path.exists(pt) and os.path.exists(meta_f) and os.path.exists(sc_f)):
            return None

        with open(meta_f, "r") as f:
            meta = json.load(f)

        # 构建与训练时一致的小 LSTM 结构
        class _TinyLSTM(nn.Module):
            def __init__(self, in_dim=1, hidden=meta["hidden_size"], layers=meta["num_layers"], drop=meta["dropout"]):
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
                out, h = self.lstm(x, h)
                last = out[:, -1, :]
                y = self.head(last)
                return y, h

        device = self.device
        model = _TinyLSTM().to(device)
        state = torch.load(pt, map_location=device)
        model.load_state_dict(state)
        model.eval()

        npz = np.load(sc_f)
        a_min = float(npz["a_min"])
        scale = float(npz["scale"])

        def _fwd(x):
            return (x - a_min) / max(1e-8, scale)

        def _inv(x):
            return x * max(1e-8, scale) + a_min

        scaler = type("TmpScaler", (), {})()
        scaler.fwd, scaler.inv = _fwd, _inv

        use_log = bool(meta.get("use_log", meta.get("log_price", True)))
        return model, scaler, use_log

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
