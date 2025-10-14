from __future__ import annotations
from typing import List
import numpy as np
from datetime import datetime, timezone

def cosine(a: List[float], b: List[float]) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def recency_score(published_at: datetime, half_life_hours: float = 24.0) -> float:
    # 兼容 naive/aware：naive 视为 UTC
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    
    # 指数衰减新鲜度，半衰期默认24h
    now = datetime.now(timezone.utc)
    dt_hours = max((now - published_at).total_seconds() / 3600.0, 0.0)
    import math
    return math.exp(-math.log(2) * dt_hours / half_life_hours)
