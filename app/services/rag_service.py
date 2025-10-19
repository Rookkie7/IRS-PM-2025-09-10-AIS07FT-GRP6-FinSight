# app/services/rag_service.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid
import httpx
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.adapters.db.database_client import get_mongo_db
from app.config import settings

def _split_csv(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

class RagService:
    """
    Minimal RAG service that proxies chat to a locally deployed RagFlow (OpenAI-compatible) backend.

    Responsibilities:
      - Create a chat session in RagFlow when no session_id is provided
      - Send a non-streaming completion request to RagFlow
      - Normalize the response into {session_id, answer, citations}
      - Fetch simple message history from RagFlow (best-effort)
    """

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.db = db
        self.base_url = (base_url or settings.RAGFLOW_BASE_URL or "http://localhost").rstrip("/")
        self.api_key = api_key or settings.RAGFLOW_API_KEY or ""
        self.model = model or getattr(settings, "RAGFLOW_MODEL", None) or "model"
        self.timeout = timeout or getattr(settings, "RAGFLOW_TIMEOUT", 15.0)

    # ------------------------
    # Public API
    # ------------------------
    async def chat(
        self,
        *,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        # 1) Ensure a chat session exists
        chat_id = session_id or await self._ensure_chat_session(user_id=user_id)

        # 2) Call RagFlow completion API (non-streaming)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": question},
            ],
            "stream": False,
        }

        url = f"{self.base_url}/api/v1/chats_openai/{chat_id}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=self._headers(), json=payload)
        except httpx.ReadTimeout as e:
            raise TimeoutError(str(e))
        except httpx.HTTPError as e:
            raise RuntimeError(f"RagFlow HTTP error: {e}")

        data = self._parse_response(resp, chat_id)
        return data

    async def history(self, session_id: str) -> List[Dict[str, Any]]:
        if not session_id:
            return []

        # Best-effort: try RagFlow messages endpoint. If it doesn't exist, return empty list.
        url = f"{self.base_url}/api/v1/chats_openai/{session_id}/messages"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, headers=self._headers())
        except httpx.ReadTimeout:
            return []
        except httpx.HTTPError:
            return []

        if resp.status_code == 403:
            # Propagate as permission error for the router to translate to 403
            try:
                j = resp.json()
                msg = j.get("message") or j.get("error") or "permission denied"
            except Exception:
                msg = "permission denied"
            raise PermissionError(msg)

        if resp.status_code >= 400:
            # Unknown schema — keep minimal implementation silent
            return []

        try:
            j = resp.json()
        except Exception:
            return []

        # Normalize: expect a list of {role, content, created_at}
        out: List[Dict[str, Any]] = []
        if isinstance(j, list):
            for it in j:
                if not isinstance(it, dict):
                    continue
                role = str(it.get("role", ""))
                content = str(it.get("content", ""))
                ts = it.get("created_at") or it.get("ts")
                out.append({"role": role, "content": content, "ts": ts})
        elif isinstance(j, dict) and "messages" in j and isinstance(j["messages"], list):
            for it in j["messages"]:
                if not isinstance(it, dict):
                    continue
                role = str(it.get("role", ""))
                content = str(it.get("content", ""))
                ts = it.get("created_at") or it.get("ts")
                out.append({"role": role, "content": content, "ts": ts})
        return out

    # ------------------------
    # Internals
    # ------------------------
    async def _ensure_chat_session(self, *, user_id: Optional[str]) -> str:
        """
        Create a chat assistant via POST /api/v1/chats and return its id.
        Response schema per docs:
        { "code": 0, "data": { "id": "<chat_id>", ... } }
        """
        base = self.base_url.rstrip("/")
        url = f"{base}/api/v1/chats"

        # 组装 body：name / dataset_ids / llm（最少只放 model_name 就能跑）
        name_prefix = getattr(settings, "RAGFLOW_CHAT_NAME_PREFIX", "chat") or "chat"
        name = f"{name_prefix}-{user_id or 'anon'}-{uuid.uuid4().hex[:6]}"
        dataset_ids = _split_csv(getattr(settings, "RAGFLOW_DATASET_IDS", ""))

        payload = {
            "name": name,
        }
        if dataset_ids:
            payload["dataset_ids"] = dataset_ids
        if self.model:
            payload["llm"] = {"model_name": self.model}

        async def do_create(p: dict) -> Optional[str]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=self._headers(), json=p)

            # 授权问题
            if resp.status_code == 403:
                try:
                    j = resp.json()
                    msg = j.get("message") or j.get("error") or "forbidden"
                except Exception:
                    msg = "forbidden"
                raise PermissionError(msg)

            # 其他 http 错误
            if resp.status_code >= 400:
                raise RuntimeError(f"RagFlow create-chat http {resp.status_code}: {resp.text}")

            try:
                j = resp.json()
            except Exception as e:
                raise RuntimeError(f"Invalid RagFlow create-chat response: {e}")

            # 文档规范：code == 0 才成功
            code = j.get("code")
            if code == 0 and isinstance(j.get("data"), dict):
                cid = j["data"].get("id")
                if cid:
                    return str(cid)
                # 防御：某些实现可能用 chat_id 字段
                cid = j["data"].get("chat_id")
                if cid:
                    return str(cid)

            # 若重复名称：{"code":102, "message":"Duplicated chat name ..."}
            if code == 102:
                return None  # 交由外层加后缀重试

            # 其它 code 非 0，抛错更直观
            raise RuntimeError(f"RagFlow create-chat failed: {j}")

        # 第一次尝试
        cid = await do_create(payload)
        if cid:
            return cid

        # 名称冲突则加后缀重试一次
        payload["name"] = f"{name}-r{uuid.uuid4().hex[:4]}"
        cid = await do_create(payload)
        if cid:
            return cid

        raise RuntimeError("RagFlow create-chat response missing 'data.id'")

    def _parse_response(self, resp: httpx.Response, chat_id: str) -> Dict[str, Any]:
        # Permission issues from RagFlow (e.g., You don't own the chat)
        if resp.status_code == 403:
            try:
                j = resp.json()
                msg = j.get("message") or j.get("error") or "forbidden"
            except Exception:
                msg = "forbidden"
            raise PermissionError(msg)

        if resp.status_code >= 500:
            raise RuntimeError(f"RagFlow server error: {resp.status_code}")

        try:
            j = resp.json()
        except Exception as e:
            raise RuntimeError(f"Invalid JSON from RagFlow: {e}")

        # Two schemas supported:
        # 1) RagFlow native: { answer: str, citations: [...] }
        # 2) OpenAI compat: { choices: [{ message: { role, content } }] }
        answer: str = ""
        citations: List[Dict[str, Any]] = []

        if isinstance(j, dict):
            if "answer" in j:
                answer = str(j.get("answer") or "")
                raw_cites = j.get("citations")
                if isinstance(raw_cites, list):
                    citations = raw_cites  # pass-through
            elif "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                choice0 = j["choices"][0]
                if isinstance(choice0, dict):
                    msg = choice0.get("message") or {}
                    if isinstance(msg, dict):
                        answer = str(msg.get("content") or "")

        return {
            "session_id": chat_id,
            "answer": answer,
            "citations": citations,
        }

    def _headers(self) -> Dict[str, str]:
        hdrs = {"Content-Type": "application/json"}
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        return hdrs


# Optional factory for DI frameworks that prefer to build services here.
async def get_rag_service_instance() -> RagService:
    db = get_mongo_db()
    return RagService(db)