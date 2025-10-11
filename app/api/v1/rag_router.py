from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from app.services.rag_service import RagService
from app.deps import get_rag_service

router = APIRouter(prefix="/rag", tags=["rag"])

class RagAnswerIn(BaseModel):
    query: Optional[str] = Field(None, description="自然语言问题")
    query_vector: Optional[List[float]] = Field(None, description="32维查询向量")
    top_k: int = 5
    temperature: float = 0.2
    max_tokens: int = 512
    filters: Optional[dict] = None

@router.post("/answer")
async def rag_answer(payload: RagAnswerIn, svc: RagService = Depends(get_rag_service)):
    try:
        res = await svc.answer(
            query_text=payload.query,
            query_vector=payload.query_vector,
            k=payload.top_k,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            filters=payload.filters,
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))