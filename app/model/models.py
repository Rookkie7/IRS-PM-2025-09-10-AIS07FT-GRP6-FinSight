from pydantic import BaseModel, ConfigDict, Field, EmailStr
from typing import List, Optional
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
import numpy as np
import json

class EmbeddingVector(BaseModel):
    dim: int
    values: List[float]

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

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

class UserPublic(BaseModel):
    id: str
    email: str
    username: str
    created_at: datetime
    profile: Dict[str, Any] = {}
    embedding: Optional[list[float]] = None  # 或 EmbeddingVector

class UserProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # 允许 ORM->Pydantic
    user_id: str
    profile_vector_20d: str = None
    industry_preferences: List[float] = []
    investment_preferences: List[float] = []
    created_at: datetime
    updated_at: datetime

class RegisterResponse(BaseModel):
    user: UserPublic
    profile: Optional[UserProfileOut] = None

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

Base = declarative_base()

class StockVector(Base):
    __tablename__ = "stock_vectors"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, index=True)
    name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(150))
    # 20维向量
    vector_20d = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_vector_20d(self):
        """获取20维向量"""
        return np.array(json.loads(self.vector_20d))

    def set_vector_20d(self, vector):
        """设置20维向量"""
        self.vector_20d = json.dumps(vector.tolist() if isinstance(vector, np.ndarray) else vector)


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String(50), primary_key=True, index=True)

    # 20维用户画像向量
    profile_vector_20d = Column(Text)
    # 详细的画像数据
    industry_preferences = Column(JSON)  # 11维行业偏好
    investment_preferences = Column(JSON)  # 9维投资偏好
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_profile_vector_20d(self):
        """获取20维用户画像向量"""
        return np.array(json.loads(self.profile_vector_20d))

    def set_profile_vector_20d(self, vector):
        """设置20维用户画像向量"""
        self.profile_vector_20d = json.dumps(vector.tolist() if isinstance(vector, np.ndarray) else vector)

    def build_vector_from_components(self):
        """从组件构建20维向量"""
        components = []

        # 1. 行业偏好 (11维)
        components.extend(self.industry_preferences or [0.5] * 11)

        # 2. 投资偏好 (9维)
        components.extend(self.investment_preferences or [0.5] * 9)

        vector = np.array(components)
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

# MongoDB模型保持为普通类
class StockRawData:
    """MongoDB中存储的股票原始数据"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.basic_info: Dict[str, Any] = {}
        self.financials: Dict[str, Any] = {}
        self.historical_data: Dict[str, Any] = {}
        self.descriptions: Dict[str, str] = {}  # 删除news字段
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'basic_info': self.basic_info,
            'financials': self.financials,
            'historical_data': self.historical_data,
            'descriptions': self.descriptions,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StockRawData':
        stock = cls(data['symbol'])
        stock.basic_info = data.get('basic_info', {})
        stock.financials = data.get('financials', {})
        stock.historical_data = data.get('historical_data', {})
        stock.descriptions = data.get('descriptions', {})
        stock.created_at = data.get('created_at', datetime.utcnow())
        stock.updated_at = data.get('updated_at', datetime.utcnow())
        return stock


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