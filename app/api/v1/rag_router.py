from fastapi import APIRouter, Depends
from app.services.rag_service import RagService
from app.deps import get_rag_service

router = APIRouter(prefix="/rag", tags=["rag"])

@router.post("/answer")
async def rag_answer(payload: dict, svc: RagService = Depends(get_rag_service)):
    ...