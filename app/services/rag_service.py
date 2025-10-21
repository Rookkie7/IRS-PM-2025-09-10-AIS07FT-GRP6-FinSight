# app/services/rag_service.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid
import httpx
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.adapters.db.database_client import get_mongo_db
from app.adapters.db.rag_conversation_repo import RagConversationRepo
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
        repo: Optional[RagConversationRepo] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        history_window: Optional[int] = None,
    ) -> None:
        self.db = db
        self.repo = repo or RagConversationRepo()
        self.base_url = (base_url or settings.RAGFLOW_BASE_URL or "http://localhost").rstrip("/")
        self.api_key = api_key or settings.RAGFLOW_API_KEY or ""
        self.model = model or getattr(settings, "RAGFLOW_MODEL", None) or "deepseek-r1:8b"
        self.timeout = timeout or getattr(settings, "RAGFLOW_TIMEOUT", 15.0)
        # 一次请求带入 RagFlow 的历史轮数（只取最近 N 条）
        self.history_window = history_window or getattr(settings, "RAG_HISTORY_WINDOW", 20)

    def _build_prompt_config(
        self,
        *,
        prompt_text: Optional[str] = None,
        top_n: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        keywords_similarity_weight: Optional[float] = None,
        show_quote: Optional[bool] = None,
        refine_multiturn: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        返回 RagFlow /api/v1/chats 的 prompt 配置对象。
        这些键名与 RagFlow 返回的结构一致（empty_response / opener 可按需设置）。
        """
        # 选择优先级：调用参数 > settings 默认 > 合理兜底
        return {
            "prompt_type": "simple",  # 如 UI 使用 simple prompt
            "empty_response": "Sorry! No relevant content was found in the knowledge base!",
            "opener": "Hi! I'm your assistant. What can I do for you?",
            "prompt": prompt_text
                or settings.RAGFLOW_PROMPT_TEXT
                or (
                    "You are an intelligent assistant. Summarize the content of the knowledge base to answer the question. "
                    "When all knowledge base content is irrelevant, your answer must include: "
                    "\"The answer you are looking for is not found in the knowledge base!\" "
                    "Answers need to consider chat history.\nHere is the knowledge base:\n{knowledge}\nThe above is the knowledge base."
                ),
            "top_n": top_n if top_n is not None else settings.RAGFLOW_PROMPT_TOP_N,
            "similarity_threshold": (
                similarity_threshold
                if similarity_threshold is not None
                else settings.RAGFLOW_PROMPT_SIMILARITY_THRESHOLD
            ),
            "keywords_similarity_weight": (
                keywords_similarity_weight
                if keywords_similarity_weight is not None
                else settings.RAGFLOW_PROMPT_KW_WEIGHT
            ),
            "show_quote": show_quote if show_quote is not None else settings.RAGFLOW_PROMPT_SHOW_QUOTE,
            "refine_multiturn": (
                refine_multiturn
                if refine_multiturn is not None
                else settings.RAGFLOW_PROMPT_REFINE_MULTITURN
            ),
            "rerank_model": "",
            "tts": False,
            # simple 模板下建议显式给出占位符变量
            "variables": [{"key": "knowledge", "optional": False}],
        }

    async def list_datasets(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/api/v1/datasets"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, headers=self._headers())
        resp.raise_for_status()
        j = resp.json()
        if isinstance(j, dict) and j.get("code") == 0 and isinstance(j.get("data"), list):
            # 仅回传必要字段
            out = []
            for it in j["data"]:
                if not isinstance(it, dict):
                    continue
                out.append({
                    "id": it.get("id"),
                    "name": it.get("name"),
                    "description": it.get("description", ""),
                })
            return out
        # 兜底：直接返回原始 data
        return j if isinstance(j, list) else j.get("data", [])

    async def list_models(self) -> List[str]:
        # 统一尝试常见两个端点
        candidates = [
            f"{self.base_url}/api/v1/models",
            f"{self.base_url}/api/llm/models",
        ]
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            last_error = None
            for url in candidates:
                try:
                    resp = await client.get(url, headers=self._headers())
                    if resp.status_code == 404:
                        continue
                    resp.raise_for_status()
                    j = resp.json()
                    # 兼容：返回可能是 {code:0, data:[{model_name:"..."}, ...]}
                    if isinstance(j, dict) and "data" in j:
                        data = j["data"]
                    else:
                        data = j
                    names: List[str] = []
                    if isinstance(data, list):
                        for it in data:
                            if isinstance(it, str):
                                names.append(it)
                            elif isinstance(it, dict):
                                # 常见字段：model / model_name / name
                                name = it.get("model") or it.get("model_name") or it.get("name")
                                if name:
                                    names.append(str(name))
                    return sorted(set(names))
                except Exception as e:
                    last_error = e
            # 所有端点都失败
            if last_error:
                raise last_error
            return []

    # ------------------------
    # Public API
    # ------------------------
    async def chat(
            self,
            *,
            question: str,
            session_id: Optional[str] = None,
            user_id: Optional[str] = None,
            model: Optional[str] = None,
            dataset_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        # 1) 如果没传 session_id，先在 RagFlow 里创建 chat（可绑定知识库 & 模型）
        chat_id = session_id
        if not chat_id:
            chat_id = await self._ensure_chat_session(
                user_id=user_id,
                model_override=model,
                dataset_ids_override=dataset_ids
            )
            # 用 RagFlow 的 chat_id 作为我方会话 id，并在本地建档
            await self.repo.start_session(session_id=chat_id)

        # 2) 带入最近 N 条上下文 + 当前问题
        history = await self.repo.get_messages(chat_id, limit=self.history_window)
        messages: List[Dict[str, str]] = []
        for m in history:
            role = str(m.get("role", "") or "user")
            content = str(m.get("content", "") or "")
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})

        # 3) 调 RagFlow OpenAI 兼容接口
        payload = {
            "model": model or self.model,
            "messages": messages,
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

        # 4) 本地持久化对话
        try:
            await self.repo.append_message(chat_id, role="user", content=question, citations=None)
            await self.repo.append_message(chat_id, role="assistant", content=data.get("answer", ""),
                                           citations=data.get("citations", []))
        except Exception:
            pass

        return data

    async def history(self, session_id: str) -> List[Dict[str, Any]]:
        if not session_id:
            return []
        # 直接返回本地保存的历史
        return await self.repo.list_history(session_id, limit=self.history_window)

    # ------------------------
    # Internals
    # ------------------------
    async def _ensure_chat_session(
        self,
        *,
        user_id: Optional[str],
        dataset_ids: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        prompt_text: Optional[str] = None,
        # 下面这些是 prompt 的细节可选覆盖（按需传）
        top_n: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        keywords_similarity_weight: Optional[float] = None,
        show_quote: Optional[bool] = None,
        refine_multiturn: Optional[bool] = None
    ) -> str:
        """
        显式在 RagFlow 创建 chat，并把自定义 prompt 一起写进去。
        """
        base = self.base_url.rstrip("/")
        url = f"{base}/api/v1/chats"

        name_prefix = getattr(settings, "RAGFLOW_CHAT_NAME_PREFIX", "chat") or "chat"
        name = f"{name_prefix}-{user_id or 'anon'}-{uuid.uuid4().hex[:6]}"

        payload: Dict[str, Any] = {
            "name": name,
            "prompt": self._build_prompt_config(
                prompt_text=prompt_text,
                top_n=top_n,
                similarity_threshold=similarity_threshold,
                keywords_similarity_weight=keywords_similarity_weight,
                show_quote=show_quote,
                refine_multiturn=refine_multiturn
            ),
        }
        if dataset_ids:
            payload["dataset_ids"] = dataset_ids
        if (model_name or self.model):
            payload["llm"] = {"model_name": model_name or self.model}

        async def do_create(p: dict) -> Optional[str]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=self._headers(), json=p)

            if resp.status_code == 403:
                try:
                    j = resp.json()
                    msg = j.get("message") or j.get("error") or "forbidden"
                except Exception:
                    msg = "forbidden"
                raise PermissionError(msg)

            if resp.status_code >= 400:
                raise RuntimeError(f"RagFlow create-chat http {resp.status_code}: {resp.text}")

            j = resp.json()
            code = j.get("code")
            if code == 0 and isinstance(j.get("data"), dict):
                cid = j["data"].get("id") or j["data"].get("chat_id")
                if cid:
                    return str(cid)
            if code == 102:  # name 冲突
                return None
            raise RuntimeError(f"RagFlow create-chat failed: {j}")

        # 首次
        cid = await do_create(payload)
        if cid:
            return cid
        # 冲突重试
        payload["name"] = f"{name}-r{uuid.uuid4().hex[:4]}"
        cid = await do_create(payload)
        if cid:
            return cid
        raise RuntimeError("RagFlow create-chat response missing 'data.id'")

    def _parse_response(self, resp: httpx.Response, chat_id: str) -> Dict[str, Any]:
        if resp.status_code == 403:
            try:
                j = resp.json()
                msg = j.get("message") or j.get("error") or "forbidden"
            except Exception:
                msg = "forbidden"
            raise PermissionError(msg)

        if resp.status_code >= 500:
            raise RuntimeError(f"RagFlow server error: {resp.status_code}")

        j = resp.json()
        answer: str = ""
        citations: List[Dict[str, Any]] = []

        if isinstance(j, dict):
            if "answer" in j:
                answer = str(j.get("answer") or "")
                raw_cites = j.get("citations")
                if isinstance(raw_cites, list):
                    citations = raw_cites
            elif "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                choice0 = j["choices"][0]
                if isinstance(choice0, dict):
                    msg = choice0.get("message") or {}
                    if isinstance(msg, dict):
                        answer = str(msg.get("content") or "")

        return {"session_id": chat_id, "answer": answer, "citations": citations}

    def _headers(self) -> Dict[str, str]:
        hdrs = {"Content-Type": "application/json"}
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        return hdrs

async def get_rag_service_instance() -> RagService:
    db = get_mongo_db()
    repo = RagConversationRepo()
    return RagService(db, repo=repo)