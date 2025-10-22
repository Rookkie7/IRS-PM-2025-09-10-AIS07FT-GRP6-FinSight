from __future__ import annotations
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4

from app.adapters.db.database_client import get_mongo_db  # 你已有的 getter（返回 DB）
from motor.motor_asyncio import AsyncIOMotorCollection

class RagConversationRepo:
    def __init__(self) -> None:
        db = get_mongo_db()
        self.col: AsyncIOMotorCollection = db["rag_conversation"]

    async def ensure_indexes(self) -> None:
        await self.col.create_index("created_at")
        await self.col.create_index("updated_at")
        await self.col.create_index("messages.ts")

    async def start_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or uuid4().hex
        now = datetime.utcnow()
        # 如果已经存在就不重复插入
        existed = await self.col.find_one({"_id": sid}, projection={"_id": 1})
        if not existed:
            doc = {
                "_id": sid,
                "created_at": now,
                "updated_at": now,
                "messages": []  # [{role, content, citations, ts}]
            }
            await self.col.insert_one(doc)
        return sid

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        msg = {
            "role": role,  # "user" | "assistant" | "system"
            "content": content,
            "citations": citations or [],
            "ts": datetime.utcnow()
        }
        await self.col.update_one(
            {"_id": session_id},
            {
                "$push": {"messages": msg},
                "$set": {"updated_at": datetime.utcnow()}
            },
            upsert=True,  # 容错：万一没先 start 也能写进来
        )

    async def get_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        读取该会话的最近 limit 条消息，按 ts 升序返回。
        注意：数组字段无法在服务端做 sort+limit（除非改用 $slice/$filter 管道），
        这里采用客户端截取+排序：保持简单且与你 append 的顺序一致。
        """
        doc = await self.col.find_one({"_id": session_id}, projection={"messages": 1, "_id": 0})
        if not doc or "messages" not in doc:
            return []
        msgs = doc["messages"] or []

        # 兜底排序，防止未来有非顺序插入的情况
        msgs.sort(key=lambda x: x.get("ts") or datetime.min)

        # 仅返回最近 limit 条（升序）
        if limit and limit > 0:
            msgs = msgs[-limit:]

        # 规范化字段类型
        norm = []
        for m in msgs:
            norm.append({
                "role": str(m.get("role", "")),
                "content": str(m.get("content", "")),
                "citations": m.get("citations", []) or [],
                "ts": m.get("ts")
            })
        return norm

    async def list_history(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        RagService.history() 可直接调用的便捷包装。
        """
        return await self.get_messages(session_id, limit=limit)