# # app/main.py
# from __future__ import annotations
# from contextlib import asynccontextmanager

# import uvicorn
# from fastapi import FastAPI

# from app.adapters.db.database_client import init_mongo_via_ssh, init_postgres_via_ssh
# from app.adapters.db.news_repo import NewsRepo
# from app.adapters.db.user_repo import UserRepo
# from fastapi.middleware.cors import CORSMiddleware

# from app.config import settings
# from app.api.v1.stocks_router import router as stocks_router
# from app.api.v1.auth_router import router as auth_router
# from app.api.v1.news_router import router as news_router
# from app.api.v1.user_router import router as user_router
# from app.api.v1.rec_router import router as rec_router
# from app.api.v1.rag_router import router as rag_router
# from app.api.v1.forecast_router import router as forecast_router
# from app.jobs.scheduler import create_scheduler

# # —— 依赖注入：全局单例（为了让 routers 通过 app.main.svc 获取服务实例）——
# from app.adapters.embeddings.hash_embedder import HashingEmbedder
# from app.adapters.embeddings.projecting_embedder import ProjectingEmbedder

# from app.repositories.mongo_repos import MongoNewsRepo, MongoEventRepo, MongoProfileRepo
# from app.repositories.pg_profile_repo import PgProfileRepo
# from app.repositories.inmemory import InMemoryNewsRepo, InMemoryProfileRepo, InMemoryEventRepo

# from app.services.news_service import NewsService
# from app.domain.models import NewsItem
# from app.utils.news_seed import SEED_NEWS
# from app.utils.healthy import check_database_connection
# from contextlib import asynccontextmanager

# from app.core.errors import http_exception_handler, validation_exception_handler, generic_exception_handler
# from app.core.middleware import RequestContextMiddleware
# from fastapi.exceptions import RequestValidationError
# from starlette.exceptions import HTTPException as StarletteHTTPException

# try:
#     from app.adapters.embeddings.sentence_transformers_embed import LocalEmbeddingProvider
#     _HAS_ST = True
# except Exception:
#     _HAS_ST = False

# def _build_embedder():
#     """根据 settings.EMBEDDING_PROVIDER 构建嵌入器。"""
#     provider = (settings.EMBEDDING_PROVIDER or "").lower().strip()
#     if provider in ("hash", "placeholder"):
#         base = HashingEmbedder(max(64, settings.PROJECTION_DIM))
#     elif provider in ("st", "sentence-transformers", "sentence_transformers"):
#         if not _HAS_ST:
#             raise RuntimeError("EMBEDDING_PROVIDER=st 但未安装 sentence-transformers。")
#         base = LocalEmbeddingProvider(settings.ST_MODEL)  # e.g. 384
#     elif provider in ("openai", "oai"):
#         # 如果你已有 OpenAI 适配器，取消上面的 import 并启用这里
#         # return OpenAIEmbedder(model=settings.OAI_EMBED_MODEL, api_key=settings.OPENAI_API_KEY)
#         raise RuntimeError("EMBEDDING_PROVIDER=openai 尚未接入具体适配器，请实现 app/adapters/embeddings/openai_embedder.py 后再启用。")
#     # 兜底
#     else:
#         base = HashingEmbedder(dim=max(64, settings.PROJECTION_DIM))

#     # 投影总开关：只要 PROJECTION_METHOD 不是 none，就包一层
#     method = (settings.PROJECTION_METHOD or "srp").lower().strip()
#     if method != "none":
#         emb = ProjectingEmbedder(
#             base_embedder=base,
#             method=method,
#             proj_dim=settings.PROJECTION_DIM,
#             seed=settings.PROJECTION_SEED
#         )
#         return emb
#     return base
  
# def _build_repos(embedder_dim: int):
#     backend = (settings.VECTOR_BACKEND or "").lower().strip()

#     if backend in ("pgvector", "pg", "postgres"):
#         print("[RepoInit] ProfileRepo = PgProfileRepo, NewsRepo/EventRepo = Mongo")
#         news_repo = MongoNewsRepo(settings.MONGO_URI, db_name=settings.MONGO_DB)
#         ev_repo   = MongoEventRepo(settings.MONGO_URI, db_name=settings.MONGO_DB)
#         prof_repo = PgProfileRepo(settings.PG_DSN, dim=embedder_dim)
#         return news_repo, prof_repo, ev_repo

#     if backend in ("mongo", "mongodb"):
#         print("[RepoInit] ProfileRepo = MongoProfileRepo, NewsRepo/EventRepo = Mongo")
#         news_repo = MongoNewsRepo(settings.MONGO_URI, db_name=settings.MONGO_DB)
#         ev_repo   = MongoEventRepo(settings.MONGO_URI, db_name=settings.MONGO_DB)
#         prof_repo = MongoProfileRepo(settings.MONGO_URI, db_name=settings.MONGO_DB, dim=embedder_dim)
#         return news_repo, prof_repo, ev_repo
#     else:
#         print("[RepoInit] ProfileRepo = InMemoryProfileRepo, NewsRepo/EventRepo = InMemory")
#         # 兜底：全部内存
#         return InMemoryNewsRepo(), InMemoryProfileRepo(dim=embedder_dim), InMemoryEventRepo()

# embedder = _build_embedder()
# news_repo, prof_repo, ev_repo = _build_repos(
#     embedder_dim=getattr(embedder, "dim", settings.DEFAULT_VECTOR_DIM)
# )

# svc = NewsService(news_repo, prof_repo, ev_repo, embedder)
# sched = None

# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     global sched

# #     # 可选：启动时导入种子数据
# #     try:
# #         svc.ingest([NewsItem(**n) for n in SEED_NEWS])
# #     except Exception:
# #         pass

# #     # —— 启动调度器（仅 dev/DEBUG）——
# #     # if settings.ENV.lower() == "dev" or settings.DEBUG:
# #     #     try:
# #     #         sched = create_scheduler(app, news_repo=news_repo, embedder=embedder)
# #     #         sched.start()
# #     #         print("[Scheduler] started with cron jobs")
# #     #     except Exception as e:
# #     #         print(f"[Scheduler] failed to start: {e}")

# #     # —— 启动调度器（仅当 ENV=dev/DEBUG 且 ENABLE_SCHEDULER=1）——
# #     if (settings.ENV.lower() == "dev" or settings.DEBUG) and getattr(settings, "ENABLE_SCHEDULER", 0):
# #         try:
# #             sched = create_scheduler(app, news_repo=news_repo, embedder=embedder)
# #             sched.start()
# #             print("[Scheduler] started with cron jobs")
# #         except Exception as e:
# #             print(f"[Scheduler] failed to start: {e}")
# #     else:
# #         print("[Scheduler] disabled (set ENABLE_SCHEDULER=1 to enable)")
# #     # 应用运行中
# #     yield

# #     # —— 关闭调度器 —— 
# #     if sched is not None:
# #         try:
# #             sched.shutdown(wait=False)
# #             print("[Scheduler] shutdown ok")
# #         except Exception as e:
# #             print(f"[Scheduler] shutdown error: {e}")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Lifespan context"""
#     async with init_mongo_via_ssh(), init_postgres_via_ssh():
#         # 启动阶段
#         user_repo = UserRepo()
#         news_repo = NewsRepo()
#         await user_repo.ensure_indexes()
#         print("✅ MongoDB indexes ensured at startup.")
#         # 交回控制权，开始处理请求
#         yield
#         # 关闭阶段（需要额外清理就放这里）
#         print("🛑 App shutting down... (cleanup if needed)")


# def create_app() -> FastAPI:
#     """
#     构建 FastAPI 应用，加载路由、中间件、事件处理等
#     """
#     app = FastAPI(
#         title=settings.APP_NAME,
#         version="1.0.0",
#         description="Finsight Backend APIs",
#         lifespan=lifespan,
#     )

#     # 中间件
#     # app.add_middleware(RequestContextMiddleware)

#      # 配置CORS
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],  # 生产环境应该限制具体域名
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )
    
#     # 全局异常处理
#     app.add_exception_handler(StarletteHTTPException, http_exception_handler)
#     app.add_exception_handler(RequestValidationError, validation_exception_handler)
#     app.add_exception_handler(Exception, generic_exception_handler)

#     # 注册路由
#     app.include_router(auth_router)
#     app.include_router(user_router)
#     app.include_router(news_router)
#     app.include_router(rec_router)
#     app.include_router(rag_router)
#     app.include_router(forecast_router)
#     app.include_router(user_router)

#     # # 调试与维护路由（仅 dev/DEBUG）
#     # if settings.ENV.lower() == "dev" or settings.DEBUG:
#     #     from app.api.v1.debug_router import router as debug_router
#     #     app.include_router(debug_router)


#     @app.get("/")
#     async def root():
#         """
#         根路径
#         """
#         db_status = check_database_connection()
#         return {
#             "message": "股票推荐系统 API",
#             "status": "running",
#             "version": "1.0.0",
#             "database_status": db_status,
#             "endpoints": {
#                 "文档": "/docs",
#                 "健康检查": "/health",
#                 "股票数据": "/api/stocks",
#                 "用户管理": "/api/users"
#             }
#         }

#     @app.get("/health")
#     async def health_check():
#         """
#         健康检查端点
#         """
#         db_status = check_database_connection()
#         return {
#             "status": "healthy",
#             "service": "stock_recommendation",
#             "database": db_status,
#             "timestamp": "2024-01-01T00:00:00Z"  # 实际应该用datetime
#         }

#     @app.get("/debug/status")
#     async def debug_status():
#         """
#         调试状态检查
#         """
#         db_status = check_database_connection()
#         return {
#             "status": "running",
#             "service": "stock_recommendation",
#             "version": "1.0.0",
#             "database": db_status,
#         }

#     return app


# app = create_app()

# if __name__ == "__main__":
#     uvicorn.run(
#         "app.main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True  # 开发模式下自动重载
#     )
    


# app/main.py
from __future__ import annotations
from contextlib import asynccontextmanager

import re
from typing import Optional
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.adapters.db.database_client import init_mongo_via_ssh, init_postgres_sync, get_mongo_db
from app.adapters.db.news_repo import NewsRepo, EventRepo
from app.adapters.db.user_repo import UserRepo
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

# —— 路由 —— #
from app.api.v1.auth_router import router as auth_router
from app.api.v1.user_router import router as user_router
from app.api.v1.news_router import router as news_router
from app.api.v1.rec_router import router as rec_router
from app.api.v1.rag_router import router as rag_router
from app.api.v1.forecast_router import router as forecast_router

# —— 异常与中间件 —— #
from app.core.errors import (
    http_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
)

# —— 你的“推荐系统”依赖注入体系（保持原样）—— #
from app.adapters.embeddings.hash_embedder import HashingEmbedder
from app.adapters.embeddings.projecting_embedder import ProjectingEmbedder
from app.repositories.pg_profile_repo import PgProfileRepo
from app.services.news_service import NewsService

import logging
logging.getLogger("app.repos.mongo_news").setLevel(logging.WARNING)
logging.getLogger("app.ingest").setLevel(logging.WARNING)
logging.getLogger("app.rec.debug").setLevel(logging.WARNING)


try:
    from app.adapters.embeddings.sentence_transformers_embed import LocalEmbeddingProvider
    _HAS_ST = True
except Exception:
    _HAS_ST = False


def _build_embedder():
    """根据 settings.EMBEDDING_PROVIDER 构建嵌入器。"""
    provider = (settings.EMBEDDING_PROVIDER or "").lower().strip()
    if provider in ("hash", "placeholder"):
        base = HashingEmbedder(max(64, settings.PROJECTION_DIM))
    elif provider in ("st", "sentence-transformers", "sentence_transformers"):
        if not _HAS_ST:
            raise RuntimeError("EMBEDDING_PROVIDER=st 但未安装 sentence-transformers。")
        base = LocalEmbeddingProvider(settings.ST_MODEL)  # e.g., 384
    elif provider in ("openai", "oai"):
        raise RuntimeError("EMBEDDING_PROVIDER=openai 尚未接入具体适配器。")
    else:
        base = HashingEmbedder(dim=max(64, settings.PROJECTION_DIM))

    method = (settings.PROJECTION_METHOD or "srp").lower().strip()
    if method != "none":
        return ProjectingEmbedder(
            base_embedder=base,
            method=method,
            proj_dim=settings.PROJECTION_DIM,
            seed=settings.PROJECTION_SEED,
        )
    return base

# —— 全局单例（仅占位；等隧道建立后在 lifespan 里初始化）——
embedder = _build_embedder()
news_repo = None
prof_repo = None
ev_repo = None
svc = None

def get_service():
    return svc

# 事件仓库：用 get_mongo_db() 做一个轻量适配（避免使用固定 URI）
class EventRepoViaClient:
    def __init__(self):
        self._col = get_mongo_db()["events"]
    async def add(self, ev):
        d = ev.model_dump() if hasattr(ev, "model_dump") else dict(ev)
        await self._col.insert_one(d)

    async def all(self, limit=200):
        cur = self._col.find({}).sort("ts", -1).limit(int(limit))
        return [doc async for doc in cur]
    def ping_detail(self):
        try:
            # 简单 ping：取 serverStatus
            get_mongo_db().command("ping")
            return True, None
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    # ✅ 新增：取最近 N 小时内用户交互过的 news_id 集合（click/like/bookmark 均计入）
    def recent_news_ids(self, user_id: str, since_hours: int = 72) -> set[str]:
        import datetime as dt
        db = get_mongo_db()
        col = db["events"]
        since = dt.datetime.utcnow() - dt.timedelta(hours=int(since_hours))
        q = {"user_id": user_id, "ts": {"$gte": since}}
        proj = {"_id": 0, "news_id": 1}
        ids = set()
        for d in col.find(q, proj):
            nid = d.get("news_id")
            if nid:
                ids.add(nid)
        return ids
     
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context"""
    async with init_mongo_via_ssh(), init_postgres_sync():
        # 启动阶段
        user_repo = UserRepo()
        await user_repo.ensure_indexes()
        print("✅ MongoDB indexes ensured at startup.")

        global news_repo, prof_repo, ev_repo, svc, embedder
        # ✅ 拼接“隧道后的本地端口”URI（不依赖 .env 的 MONGO_URI）
        mongo_uri_local = f"mongodb://{settings.LOCAL_MONGO_HOST}:{settings.LOCAL_MONGO_PORT}"
        news_repo = NewsRepo(mongo_uri_local, db_name=settings.MONGO_DB)

        # ✅ PG 仍用本地映射端口（init_postgres_sync 建好后可用）
        dsn_local = (
            f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.LOCAL_PG_HOST}:{settings.LOCAL_PG_PORT}/{settings.POSTGRES_DB}"
        )
        prof_repo = PgProfileRepo(dsn_local, dim=getattr(embedder, "dim", settings.DEFAULT_VECTOR_DIM))

        ev_repo = EventRepo(mongo_uri_local, db_name=settings.MONGO_DB)
        svc = NewsService(news_repo, prof_repo, ev_repo, embedder)

        # 交回控制权，开始处理请求
        yield

def _mask_dsn(dsn: Optional[str]) -> Optional[str]:
    """postgresql://user:***@host:port/db"""
    if not dsn:
        return None
    return re.sub(r"(://[^:]+:)([^@]+)(@)", r"\1***\3", dsn)

def _mask_mongo(uri: Optional[str]) -> Optional[str]:
    """mongodb://user:***@host:port/..."""
    if not uri:
        return None
    return re.sub(r"(://[^:]+:)([^@]+)(@)", r"\1***\3", uri)

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description="Finsight Backend APIs",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],  # 生产环境应该限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 全局异常
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # 路由（去掉重复 include）
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(news_router)
    app.include_router(rec_router)
    app.include_router(rag_router)
    app.include_router(forecast_router)

    # 根路径
    @app.get("/")
    async def root():
        return {
            "message": "股票推荐系统 API",
            "status": "running",
            "version": "1.0.0",
            "endpoints": {
                "文档": "/docs",
                "健康检查": "/health",
            },
        }

    # 健康检查（返回纯字典，避免协程/异常对象）
    @app.get("/health")
    async def health_check():
        return {"ok": True}

    # 调试状态：沿用你之前那套 “embedder + 三仓库 ping”
    @app.get("/debug/status")
    async def debug_status():
        # embedder probe
        emb_ok, emb_err = True, None
        try:
            vec = embedder.embed_text("ping")
            emb_ok = isinstance(vec, list) and len(vec) == getattr(embedder, "dim", 64)
            if not emb_ok:
                emb_err = f"Unexpected embed result len={len(vec) if hasattr(vec,'__len__') else 'n/a'}"
        except Exception as e:
            emb_ok, emb_err = False, f"{type(e).__name__}: {e}"

        news_ok, news_err = getattr(news_repo, "ping_detail", lambda: (True, None))()
        prof_ok, prof_err = getattr(prof_repo, "ping_detail", lambda: (True, None))()
        ev_ok,   ev_err   = getattr(ev_repo,   "ping_detail", lambda: (True, None))()

        return {
            "ok": emb_ok and news_ok and prof_ok and ev_ok,
            "stores": {
                "news":   {"ok": news_ok, "error": news_err},
                "profile":{"ok": prof_ok, "error": prof_err},
                "event":  {"ok": ev_ok,   "error": ev_err},
            },
            "embedder": {"ok": emb_ok, "dim": getattr(embedder, "dim", None), "error": emb_err},
        }
    

    @app.get("/debug/repos")
    async def debug_repos():
        """
        查看当前进程实际持有的仓库实现、嵌入器维度、以及（遮罩后的）连接串。
        用于确认“我连的就是远端，而不是本地”。
        """
        return {
            "vector_backend": (settings.VECTOR_BACKEND or "").lower(),
            "embedder": {
                "class": type(embedder).__name__,
                "dim": getattr(embedder, "dim", None),
            },
            "repos": {
                "news_repo": type(news_repo).__name__,
                "profile_repo": type(prof_repo).__name__,
                "event_repo": type(ev_repo).__name__,
            },
            "connections": {
                "mongo_uri": _mask_mongo(settings.MONGO_URI),
                "mongo_db": settings.MONGO_DB,
                "pg_dsn": _mask_dsn(settings.PG_DSN),
            },
        }

    @app.post("/debug/profile/echo")
    async def debug_profile_echo(user_id: str = "cnf_probe_local"):
        """
        单次请求内完成：读取三路向量 -> 用固定“假新闻向量”做一次更新 -> 再读回结果。
        便于快速验证 PG 写入是否命中“你以为的那套库”（尤其在有 SSH 隧道时）。
        """
        # 取一次
        before = prof_repo.get_user_vectors(user_id)
        # 组织一次“假的”点击增量（64维单位向量 + 20维轻微抬升）
        sem64 = [0.0] * 64
        sem64[0] = 1.0  # 简单地把第0位置1
        prof20 = [0.0] * 20
        prof20[3] = 0.2  # 比如“Healthcare”维度轻微抬升

        # 写一次
        prof_repo.update_user_vectors_from_event(
            user_id=user_id,
            news_sem=sem64,
            news_prof=prof20,
            weight=1.0,
            alpha_short=0.4,
            alpha_long=0.1,
            alpha_prof=0.15,
        )
        # 再读一次
        after = prof_repo.get_user_vectors(user_id)

        def _norm(x):
            return None if not x else round(sum(t*t for t in x) ** 0.5, 6)

        return {
            "user_id": user_id,
            "before": {
                "short_norm": _norm(before.get("short")),
                "long_norm":  _norm(before.get("long")),
                "prof20_sum": round(sum(before.get("prof20", []) or []), 6),
            },
            "after": {
                "short_norm": _norm(after.get("short")),
                "long_norm":  _norm(after.get("long")),
                "prof20_sum": round(sum(after.get("prof20", []) or []), 6),
            }
        }

    # —— 调高关键模块日志 —— #
    import logging
    for name in [
        "app.services.news_service",
        "app.services.ingest_pipeline",
        "app.adapters.fetchers.marketaux_fetcher",
        "app.adapters.db.news_repo",
    ]:
        logging.getLogger(name).setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )