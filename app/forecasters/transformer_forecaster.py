# app/forecasters/transformer_forecaster.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class TransConfig:
    lookback: int = 96
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    epochs: int = 40
    lr: float = 1e-3
    batch_size: int = 64

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1,max_len,d]
    def forward(self, x):  # [B,L,d]
        L = x.size(1)
        return x + self.pe[:, :L, :]

class _TransformerHead(nn.Module):
    def __init__(self, cfg: TransConfig):
        super().__init__()
        self.inp = nn.Linear(1, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.nhead,
                                               dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
                                               batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.pos = _PositionalEncoding(cfg.d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = None  # 延迟创建以匹配 horizon

    def forward(self, x, tgt_len: int):
        h = self.inp(x)        # [B,L,d]
        h = self.pos(h)
        h = self.enc(h)        # [B,L,d]
        h = h.transpose(1,2)   # [B,d,L]
        h = self.pool(h).squeeze(-1)  # [B,d]
        if (self.proj is None) or (self.proj.out_features != tgt_len):
            self.proj = nn.Linear(h.size(1), tgt_len).to(h.device)
        y = self.proj(h)       # [B,T]
        return y.unsqueeze(-1)

class TransformerForecaster:
    name = "transformer"

    def __init__(self,
                 device: str = "cpu",
                 lookback: int = 96,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 epochs: int = 40,
                 lr: float = 1e-3,
                 batch_size: int = 64):
        self.cfg = TransConfig(lookback, d_model, nhead, num_layers, dim_feedforward,
                               dropout, epochs, lr, batch_size)
        self.device = torch.device(device)
        self.model = _TransformerHead(self.cfg).to(self.device)
        self.mu = None; self.sigma = None

    @staticmethod
    def _make_windows(arr: np.ndarray, L: int, T: int):
        X, Y = [], []
        for i in range(len(arr) - L - T + 1):
            X.append(arr[i:i+L])
            Y.append(arr[i+L:i+L+T])
        X = np.array(X)[:, :, None]; Y = np.array(Y)[:, :, None]
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
            y = self.model(xb, horizon)
        y = y.cpu().numpy()[0,:,0]
        return (y*self.sigma + self.mu).tolist()
