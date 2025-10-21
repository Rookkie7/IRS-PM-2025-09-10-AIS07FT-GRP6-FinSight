from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class SingleQuery(BaseModel):
    question: str
    session_id: Optional[str] = None  # 新增，可选：用于复用会话

class SingleAnswer(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    session_id: str  # 新增：返回会话 id，便于后续多轮