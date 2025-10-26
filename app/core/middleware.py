from __future__ import annotations
import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        start = time.time()
        try:
            response = await call_next(request)
        finally:
            dur_ms = int((time.time() - start) * 1000)
            path = request.url.path
            method = request.method
            # 简单 stdout 日志（你也可以接到 logging）
            print(f"[REQ] {req_id} {method} {path} {dur_ms}ms")
        # 回传响应头，便于排查
        response.headers["X-Request-ID"] = req_id
        return response
