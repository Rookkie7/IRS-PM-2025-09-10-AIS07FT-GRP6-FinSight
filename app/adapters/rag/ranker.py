from typing import List, Dict, Tuple
from datetime import datetime, timezone

class SimpleRanker:
    """
    例：综合相似度 + 时间衰减 的简易重排序
    """
    def __init__(self, alpha: float = 0.8, half_life_hours: float = 72):
        self.alpha = alpha
        self.half_life = half_life_hours

    def rerank(self, hits: List[Tuple[str, float]], docs: Dict[str, dict], top_k: int = 5) -> List[Tuple[str, float]]:
        now = datetime.now(timezone.utc).timestamp()
        scored = []
        for _id, sim in hits:
            doc = docs.get(_id) or {}
            ts = (doc.get("published_at") or doc.get("created_at"))
            if ts:
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace("Z","+00:00")).timestamp()
                    except Exception:
                        ts = now
                elif isinstance(ts, datetime):
                    ts = ts.timestamp()
            else:
                ts = now
            age_h = max(0, (now - ts) / 3600.0)
            decay = 0.5 ** (age_h / self.half_life)  # 半衰期衰减
            final = self.alpha * sim + (1 - self.alpha) * decay
            scored.append((_id, final))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]