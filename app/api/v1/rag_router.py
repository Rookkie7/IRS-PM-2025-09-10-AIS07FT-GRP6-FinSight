from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from app.deps import get_rag_service

router = APIRouter(prefix="/rag", tags=["rag"])

class QARequest(BaseModel):
    question: str

@router.post("/ask")
async def ask(req: QARequest):
    answer = await ask_fastgpt(req.question)
    return {"answer": answer}