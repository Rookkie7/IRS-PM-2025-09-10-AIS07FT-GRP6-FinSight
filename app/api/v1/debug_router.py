from __future__ import annotations
import re
from fastapi import APIRouter, Depends, HTTPException
from app.config import settings
from app.services.news_service import NewsService

def get_service() -> NewsService:
    from app.main import svc
    return svc

router = APIRouter(prefix="", tags=["debug"])

def _guard():
    if settings.ENV.lower() != "dev" and not settings.DEBUG:
        raise HTTPException(status_code=403, detail="Debug endpoints disabled in this environment.")

def _mask_dsn(dsn: str | None) -> str | None:
    if not dsn: return None
    # 隐藏密码：postgresql://user:***@host:port/db
    return re.sub(r"(://[^:]+:)([^@]+)(@)", r"\1***\3", dsn)

def _mask_mongo(uri: str | None) -> str | None:
    if not uri: return None
    # mongodb://user:***@host:port/...
    return re.sub(r"(://[^:]+:)([^@]+)(@)", r"\1***\3", uri)

@router.get("/debug/repos")
def debug_repos(service: NewsService = Depends(get_service)):
    _guard()
    return {
        "news_repo": type(service.news_repo).__name__,
        "profile_repo": type(service.prof_repo).__name__,
        "event_repo": type(service.ev_repo).__name__,
        "embedder": type(service.embedder).__name__,
        "embedder_dim": getattr(service.embedder, "dim", None),
    }

@router.get("/debug/status")
def debug_status(service: NewsService = Depends(get_service)):
    _guard()
    # embedder probe
    emb_ok, emb_err = True, None
    try:
        vec = service.embedder.embed_text("ping")
        emb_ok = isinstance(vec, list) and len(vec) == getattr(service.embedder, "dim", 32)
        if not emb_ok:
            emb_err = f"Unexpected embed result shape: {type(vec)} len={len(vec) if hasattr(vec,'__len__') else 'n/a'}"
    except Exception as e:
        emb_ok, emb_err = False, f"{e.__class__.__name__}: {e}"

    # storage probes with detailed error
    news_ok, news_err = getattr(service.news_repo, "ping_detail", lambda: (True, None))()
    prof_ok, prof_err = getattr(service.prof_repo, "ping_detail", lambda: (True, None))()
    ev_ok,   ev_err   = getattr(service.ev_repo,   "ping_detail", lambda: (True, None))()

    return {
        "ok": emb_ok and news_ok and prof_ok and ev_ok,
        "stores": {
            "news":   {"ok": news_ok, "error": news_err},
            "profile":{"ok": prof_ok, "error": prof_err},
            "event":  {"ok": ev_ok,   "error": ev_err},
        },
        "embedder": {"ok": emb_ok, "dim": getattr(service.embedder, "dim", None), "error": emb_err},
        "env": {"ENV": settings.ENV, "DEBUG": settings.DEBUG},
        "connections": {
            "mongo_uri": _mask_mongo(settings.MONGO_URI),
            "pg_dsn": _mask_dsn(settings.PG_DSN),
        }
    }

@router.post("/admin/maintenance/clear_news")
def clear_news(service: NewsService = Depends(get_service)):
    _guard()
    if not hasattr(service.news_repo, "clear"):
        raise HTTPException(status_code=501, detail="Clear not supported for current news repo")
    service.news_repo.clear()
    return {"ok": True, "cleared": True}

@router.get("/debug/news/latest")
# def debug_latest_news(limit: int = 5, service: NewsService = Depends(get_service)):
#     _guard()
#     coll = getattr(service.news_repo, "coll", None)
#     if coll is None:
#         raise HTTPException(status_code=501, detail="Current news repo has no raw collection access")
#     docs = list(
#         coll.find({}, {
#             "_id": 0, "news_id": 1, "source": 1, "title": 1, "published_at": 1,
#             "tickers": 1, "topics": 1, "sentiment": 1
#         }).sort("published_at", -1).limit(int(limit))
#     )
#     return {"count": len(docs), "items": docs}

# def debug_news_latest(limit: int = 5, service: NewsService = Depends(get_service)):
#     """
#     与底层实现无关，统一走仓库接口 news_repo.latest()，
#     返回最近的新闻及关键字段，便于排查向量等。
#     """
#     try:
#         items = service.news_repo.latest(limit=limit)
#     except AttributeError:
#         # 如果某个仓库没实现 latest，就给出明确提示
#         raise HTTPException(status_code=501, detail="Current news repo does not implement latest()")

#     result = []
#     for it in items:
#         result.append({
#             "news_id": it.news_id,
#             "source": it.source,
#             "title": it.title,
#             "tickers": getattr(it, "tickers", []) or [],
#             "topics": getattr(it, "topics", []) or [],
#             "published_at": getattr(it, "published_at", None),
#             "v64_dim": len(getattr(it, "vector", []) or []),              # 语义向量长度
#             "v20_dim": len(getattr(it, "vector_prof_20d", []) or []),     # 画像向量长度
#             "has_url": bool(getattr(it, "url", "")),
#         })
#     return {"items": result}

@router.get("/debug/news/latest")
def debug_news_latest(limit: int = 5, service: NewsService = Depends(get_service)):
    _guard()

    # 优先：直接读取 raw 文档，避免模型转换遮蔽字段
    if hasattr(service.news_repo, "raw_latest_docs"):
        docs = service.news_repo.raw_latest_docs(limit)
        for d in docs:
            v = d.get("vector") or []
            p = d.get("vector_prof_20d") or []
            d["v64_dim"] = len(v)
            d["v20_dim"] = len(p)
            d["text_len"] = len(d.get("text") or "")
        return {"items": docs}

    # 兜底：退回到仓库的 latest()（如果实现了）
    if hasattr(service.news_repo, "latest"):
        items = service.news_repo.latest(limit=limit)
        out = []
        for it in items:
            v = getattr(it, "vector", []) or []
            p = getattr(it, "vector_prof_20d", []) or []
            out.append({
                "news_id": it.news_id,
                "source": it.source,
                "title": it.title,
                "tickers": getattr(it, "tickers", []) or [],
                "topics": getattr(it, "topics", []) or [],
                "published_at": getattr(it, "published_at", None),
                "text_len": len(getattr(it, "text", "") or ""),
                "v64_dim": len(v),
                "v20_dim": len(p),
                "has_url": bool(getattr(it, "url", "")),
            })
        return {"items": out}

    raise HTTPException(status_code=501, detail="Current news repo has no raw collection access nor latest()")