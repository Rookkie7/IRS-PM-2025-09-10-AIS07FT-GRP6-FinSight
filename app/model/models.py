from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class EmbeddingVector(BaseModel):
    dim: int
    values: List[float]

class News(BaseModel):
    # id: Optional[str] = None
    # title: str
    # text: str
    # source: str
    # published_at: datetime
    # tags: List[str] = []
    # labels: List[str] = []
    # embedding: Optional[EmbeddingVector] = None
    # cleaned: bool = False
    ...

class Stock(BaseModel):
    # ticker: str
    # name: str
    # sector: Optional[str] = None
    # factors: dict = {}
    # embedding: Optional[EmbeddingVector] = None
    # updated_at: datetime
    ...

class Recommendation(BaseModel):
    # user_id: str
    # tickers: List[str]
    # scores: List[float]
    # reason: Optional[str] = None
    # generated_at: datetime
    ...

class ForecastResult(BaseModel):
    # ticker: str
    # horizon_days: int
    # method: str
    # yhat: List[float]
    # generated_at: datetime
    ...