from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

# ---- 核心域模型（与文档一致：用户画像=32维；新闻含元数据/情绪/向量）----
# 画像将来可扩展为“显式维度(行业/风格/风险/主题/地域/因子) + 隐式维度”；
# 这里先用32维占位实现，接口保持稳定，便于接入Two-Tower/LightFM训练的隐向量。
# 参考 proposal 3.1/3.2/系统设计（News 与 Profile 向量空间对齐）
# （后续把 vector: List[float] 替换为 pgvector[n] 存储）.

Vector = List[float]  # 长度固定为32的向量；此MVP不强制校验长度，便于快速试跑

class UserProfile(BaseModel):
    user_id: str
    vector: Vector = Field(default_factory=lambda: [0.0]*32)  # 初始中性画像

class NewsItem(BaseModel):
    news_id: str
    title: str
    text: str
    source: str
    url: str
    published_at: datetime
    tickers: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    sentiment: float = 0.0  # [-1,1]，将来用 FinBERT: P(pos)-P(neg)
    vector: Optional[Vector] = None  # 入库前由嵌入器生成
    vector_prof_20d: List[float] = Field(default_factory=list)
    model_config = {
        "extra": "ignore"  # 允许额外字段被忽略（保留原行为）
    }
class BehaviorEvent(BaseModel):
    user_id: str
    news_id: str
    type: Literal["click","dwell","save","dislike"] = "click"
    dwell: float = 0.0      # 秒
    ts: datetime = Field(default_factory=datetime.utcnow)
