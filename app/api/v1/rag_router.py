# app/api/v1/rag_router.py
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from app.config import settings
from app.adapters.db.database_client import get_mongo_db
from app.services.rag_service import RagService, get_rag_service_instance
from motor.motor_asyncio import AsyncIOMotorDatabase

router = APIRouter(prefix="/rag", tags=["rag"])

# ---- Schemas ----
class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID；不传则新建")
    user_id: Optional[str] = Field(None, description="可选：业务用户ID")

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


@router.post("/chat", response_model=ChatResponse)
async def rag_chat(req: ChatRequest, svc: RagService = Depends(get_rag_service_instance)):
    """发起一次 RAG 对话（非流式）。
    - 如果 `session_id` 为空，服务将创建一个新的会话并返回其 ID。
    - 返回 answer 以及可选的 citations。
    """
    try:
        # 委托 RagService 完成具体逻辑（包含与 RagFlow 通讯）
        result = await svc.chat(
            question=req.question,
            session_id=req.session_id,
            user_id=req.user_id,
        )

        # 期望 result 至少包含: session_id, answer, citations
        if not isinstance(result, dict):
            raise RuntimeError("RagService.chat must return a dict")

        session_id = result.get("session_id")
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        if not session_id:
            # 由服务端保证创建/返回 session_id
            raise RuntimeError("RagService.chat did not return session_id")

        return ChatResponse(session_id=session_id, answer=answer, citations=citations)

    except HTTPException:
        # 直接透传 FastAPI HTTP 异常
        raise
    except ValueError as e:
        # 参数类错误
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        # 授权/所有权类错误（例如 RagFlow 返回 "You don't own this chat"）
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except TimeoutError as e:
        # RagFlow 超时
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e))
    except Exception as e:
        # 兜底
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"RAG chat failed: {e}")



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