# app/api/v1/rag_router.py
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from app.config import settings
from app.adapters.db.database_client import get_mongo_db
from app.services.rag_service import RagService, get_rag_service_instance
from app.adapters.db.rag_conversation_repo import RagConversationRepo

router = APIRouter(prefix="/rag", tags=["rag"])

# ---- Schemas ----
class DatasetInfo(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None

class DatasetsResponse(BaseModel):
    items: List[DatasetInfo]

class ModelsResponse(BaseModel):
    items: List[str]

class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID；不传则新建")
    user_id: Optional[str] = Field(None, description="可选：业务用户ID")
    # 新增：可选模型 & 知识库
    model: Optional[str] = None
    dataset_ids: Optional[List[str]] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[Dict[str, Any]] = []

class HistoryItem(BaseModel):
    role: str
    content: str
    ts: Optional[str] = None

class HistoryResponse(BaseModel):
    session_id: str
    turns: List[HistoryItem] = []

@router.get("/datasets", response_model=DatasetsResponse)
async def list_datasets(svc: RagService = Depends(get_rag_service_instance)):
    try:
        items = await svc.list_datasets()
        # 归一化
        norm = []
        for it in items:
            if isinstance(it, dict):
                _id = it.get("id")
                if not _id:
                    continue
                norm.append(DatasetInfo(
                    id=str(_id),
                    name=it.get("name"),
                    description=it.get("description")
                ))
        return DatasetsResponse(items=norm)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch datasets failed: {e}")

@router.get("/models", response_model=ModelsResponse)
async def list_models(svc: RagService = Depends(get_rag_service_instance)):
    try:
        items = await svc.list_models()
        return ModelsResponse(items=items)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch models failed: {e}")


@router.post("/chat", response_model=ChatResponse)
async def rag_chat(req: ChatRequest, svc: RagService = Depends(get_rag_service_instance)):
    """
    发起一次 RAG 对话（非流式）。
    - 支持前端选择的 model / dataset_ids
    - 若未提供 session_id，会在 RagFlow 创建 chat 并返回其 ID（本地也用它做会话ID）
    """
    try:
        result = await svc.chat(
            question=req.question,
            session_id=req.session_id,
            user_id=req.user_id,
            model=req.model,
            dataset_ids=req.dataset_ids
        )
        if not isinstance(result, dict):
            raise RuntimeError("RagService.chat must return a dict")

        sid = result.get("session_id")
        if not sid:
            raise RuntimeError("RagService.chat did not return session_id")

        return ChatResponse(
            session_id=sid,
            answer=result.get("answer", ""),
            citations=result.get("citations", []) or []
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG chat failed: {e}")

@router.get("/history/{session_id}", response_model=HistoryResponse)
async def rag_history(session_id: str, svc: RagService = Depends(get_rag_service_instance)):
    """查询指定会话的历史轮次。"""
    try:
        hist = await svc.history(session_id)
        # 期望 hist 为 [{"role": "user/assistant/system", "content": "...", "ts": "..."}, ...]
        turns: List[HistoryItem] = []
        if isinstance(hist, list):
            for it in hist:
                if not isinstance(it, dict):
                    continue
                turns.append(HistoryItem(
                    role=str(it.get("role", "")),
                    content=str(it.get("content", "")),
                    ts=it.get("ts") if it.get("ts") is not None else None,
                ))
        return HistoryResponse(session_id=session_id, turns=turns)

    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Fetch history failed: {e}")