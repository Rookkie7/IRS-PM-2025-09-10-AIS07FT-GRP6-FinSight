from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
from datetime import datetime

class EmbeddingVector(BaseModel):
    dim: int
    values: List[float]

class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str
    # 个人信息（将用于生成32维向量）
    full_name: Optional[str] = None
    bio: Optional[str] = None
    interests: List[str] = []
    sectors: List[str] = []  # 关注行业
    tickers: List[str] = []  # 关注股票

class UserInDB(UserBase):
    id: Optional[str] = None
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    profile: dict = {}
    embedding: Optional[EmbeddingVector] = None
    is_active: bool = True

class UserPublic(UserBase):
    id: str
    created_at: datetime
    profile: dict = {}
    embedding: Optional[EmbeddingVector] = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


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