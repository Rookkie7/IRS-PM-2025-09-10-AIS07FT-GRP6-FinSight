# app/forecasters/seq2seq_forecaster.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class Seq2SeqConfig:
    lookback: int = 64
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    epochs: int = 40
    lr: float = 1e-3
    batch_size: int = 64
    teacher_forcing: float = 0.5  # 训练期有助稳定

class _Enc(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return h, c

class _Dec(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.proj = nn.Linear(hidden, 1)
    def forward(self, x, h, c):
        out, (h, c) = self.lstm(x, (h, c))
        y = self.proj(out)
        return y, h, c

class _Seq2Seq(nn.Module):
    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.enc = _Enc(1, cfg.hidden_size, cfg.num_layers, cfg.dropout)
        self.dec = _Dec(1, cfg.hidden_size, cfg.num_layers, cfg.dropout)

    def forward(self, src, tgt_len: int, teacher_forcing: float = 0.0, y_true=None):
        # src: [B, L, 1] -> 预测未来 tgt_len 步
        h, c = self.enc(src)
        B = src.size(0)
        last = src[:, -1:, :]           # 用最后一个值启动decoder
        outs = []
        for t in range(tgt_len):
            out, h, c = self.dec(last, h, c)  # out: [B,1,1]
            outs.append(out)
            if self.training and (y_true is not None) and (torch.rand(1).item() < teacher_forcing):
                last = y_true[:, t:t+1, :].detach()
            else:
                last = out.detach()
        return torch.cat(outs, dim=1)   # [B, T, 1]

class Seq2SeqForecaster:
    name = "seq2seq"

    def __init__(self,
                 device: str = "cpu",
                 lookback: int = 64,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 epochs: int = 40,
                 lr: float = 1e-3,
                 batch_size: int = 64,
                 teacher_forcing: float = 0.5):
        self.cfg = Seq2SeqConfig(lookback, hidden_size, num_layers, dropout,
                                 epochs, lr, batch_size, teacher_forcing)
        self.device = torch.device(device)
        self.model = _Seq2Seq(self.cfg).to(self.device)
        self.mu = None
        self.sigma = None

    @staticmethod
    def _make_windows(arr: np.ndarray, L: int, T: int):
        X, Y = [], []
        for i in range(len(arr) - L - T + 1):
            X.append(arr[i:i+L])
            Y.append(arr[i+L:i+L+T])
        X = np.array(X)[:, :, None]  # [N,L,1]
        Y = np.array(Y)[:, :, None]  # [N,T,1]
        return X, Y

    def fit(self, closes: Sequence[float], horizon: int = 7):
        x = np.asarray(closes, dtype=np.float32)
        self.mu, self.sigma = float(x.mean()), float(x.std() + 1e-8)
        xn = (x - self.mu) / self.sigma

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
                pred = self.model(xb, yb.size(1), self.cfg.teacher_forcing, y_true=yb)
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
        xn = (x - self.mu) / self.sigma
        with torch.no_grad():
            xb = torch.from_numpy(xn[None,:,None]).to(self.device)
            self.model.eval()
            y = self.model(xb, horizon)  # [1,T,1]
        y = y.cpu().numpy()[0,:,0]
        return (y * self.sigma + self.mu).tolist()
