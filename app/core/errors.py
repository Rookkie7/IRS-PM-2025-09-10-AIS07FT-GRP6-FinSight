from __future__ import annotations
import traceback
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.config import settings

def _base_payload(code: str, message: str, detail: dict | None = None, trace_id: str | None = None):
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "detail": detail or {},
            "trace_id": trace_id or str(uuid.uuid4())
        }
    }

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    payload = _base_payload(
        code=f"HTTP_{exc.status_code}",
        message=exc.detail if isinstance(exc.detail, str) else "HTTP error",
        detail={"path": str(request.url)}
    )
    return JSONResponse(status_code=exc.status_code, content=payload)

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    payload = _base_payload(
        code="VALIDATION_ERROR",
        message="Invalid request parameters",
        detail={"errors": exc.errors(), "body": exc.body}
    )
    return JSONResponse(status_code=422, content=payload)

async def generic_exception_handler(request: Request, exc: Exception):
    trace_id = str(uuid.uuid4())
    detail = {"path": str(request.url)}
    if settings.DEBUG:
        detail["traceback"] = traceback.format_exc(limit=8)
        detail["type"] = exc.__class__.__name__
    payload = _base_payload(code="INTERNAL_ERROR", message="Internal server error", detail=detail, trace_id=trace_id)
    return JSONResponse(status_code=500, content=payload)
