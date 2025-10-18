# app/forecasters/dilated_cnn_forecaster.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class DCNNConfig:
    lookback: int = 64
    channels: int = 64
    n_blocks: int = 4
    kernel_size: int = 3
    dropout: float = 0.0
    epochs: int = 30
    lr: float = 2e-3
    batch_size: int = 128

class _CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, padding=self.pad, dilation=dilation)
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad != 0 else out

class _DilatedStack(nn.Module):
    def __init__(self, c, k, n_blocks, dropout):
        super().__init__()
        layers = []
        for b in range(n_blocks):
            d = 2**b
            layers += [
                _CausalConv1d(c, c, k, d), nn.ReLU(), nn.Dropout(dropout),
                _CausalConv1d(c, c, k, d), nn.ReLU(), nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: [B,C,L]
        return self.net(x)

class _DCNNSeq2Seq(nn.Module):
    def __init__(self, cfg: DCNNConfig):
        super().__init__()
        self.inp = nn.Conv1d(1, cfg.channels, kernel_size=1)
        self.stack = _DilatedStack(cfg.channels, cfg.kernel_size, cfg.n_blocks, cfg.dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B,C,1]
            nn.Flatten(),             # [B,C]
        )
        # 线性一次性输出 T 步
        self.out = None  # 延迟在 forward 里创建以适配 horizon

    def forward(self, x, tgt_len: int):
        x = x.transpose(1,2)         # [B,L,1] -> [B,1,L]
        h = self.inp(x)              # [B,C,L]
        h = self.stack(h)            # [B,C,L]
        h = self.head(h)             # [B,C]
        if (self.out is None) or (self.out.out_features != tgt_len):
            self.out = nn.Linear(h.size(1), tgt_len).to(h.device)
        y = self.out(h)              # [B,T]
        return y.unsqueeze(-1)       # [B,T,1]

class DilatedCNNForecaster:
    name = "dilated_cnn_seq2seq"

    def __init__(self,
                 device: str = "cpu",
                 lookback: int = 64,
                 channels: int = 64,
                 n_blocks: int = 4,
                 kernel_size: int = 3,
                 dropout: float = 0.0,
                 epochs: int = 30,
                 lr: float = 2e-3,
                 batch_size: int = 128):
        self.cfg = DCNNConfig(lookback, channels, n_blocks, kernel_size, dropout, epochs, lr, batch_size)
        self.device = torch.device(device)
        self.model = _DCNNSeq2Seq(self.cfg).to(self.device)
        self.mu = None; self.sigma = None

    @staticmethod
    def _make_windows(arr: np.ndarray, L: int, T: int):
        X, Y = [], []
        for i in range(len(arr) - L - T + 1):
            X.append(arr[i:i+L])
            Y.append(arr[i+L:i+L+T])
        X = np.array(X)[:, :, None]
        Y = np.array(Y)[:, :, None]
        return X, Y

    def fit(self, closes: Sequence[float], horizon: int = 7):
        x = np.asarray(closes, dtype=np.float32)
        self.mu, self.sigma = float(x.mean()), float(x.std() + 1e-8)
        xn = (x - self.mu)/self.sigma
        X, Y = self._make_windows(xn, self.cfg.lookback, horizon)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        crit = nn.MSELoss()
        best = math.inf; patience=8; bad=0; best_state=None

        self.model.train()
        for ep in range(self.cfg.epochs):
            loss_ep = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb, yb.size(1))
                loss = crit(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                loss_ep += loss.item()*xb.size(0)
            loss_ep /= len(ds)
            if loss_ep < best - 1e-6:
                best = loss_ep; bad=0
                best_state = {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
            else:
                bad += 1
            if bad >= patience: break
        if best_state: self.model.load_state_dict(best_state)

    def predict(self, recent_closes: Sequence[float], horizon: int = 7):
        assert self.mu is not None, "call fit() first"
        x = np.asarray(recent_closes[-self.cfg.lookback:], dtype=np.float32)
        xn = (x - self.mu)/self.sigma
        with torch.no_grad():
            xb = torch.from_numpy(xn[None,:,None]).to(self.device)
            self.model.eval()
            y = self.model(xb, horizon)  # [1,T,1]
        y = y.cpu().numpy()[0,:,0]
        return (y*self.sigma + self.mu).tolist()
