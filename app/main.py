# app/main.py
from contextlib import asynccontextmanager
from app.jobs.scheduler import create_scheduler

# —— 依赖注入：全局单例（为了让 routers 通过 app.main.svc 获取服务实例）——
from app.adapters.embeddings.hash_embedder import HashingEmbedder
from app.adapters.embeddings.projecting_embedder import ProjectingEmbedder  # NEW

# from app.adapters.embeddings.openai_embedder import OpenAIEmbedder

from app.repositories.mongo_repos import MongoNewsRepo, MongoEventRepo, MongoProfileRepo
from app.repositories.pg_profile_repo import PgProfileRepo
from app.repositories.inmemory import InMemoryNewsRepo, InMemoryProfileRepo, InMemoryEventRepo

from app.services.news_service import NewsService
from app.domain.models import NewsItem
from app.utils.news_seed import SEED_NEWS
from contextlib import asynccontextmanager

from app.core.errors import http_exception_handler, validation_exception_handler, generic_exception_handler
from app.core.middleware import RequestContextMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from __future__ import annotations
import uvicorn
import logging
from fastapi import FastAPI

from app.adapters.db.database_client import init_mongo_via_ssh, init_postgres_sync
from app.adapters.db.news_repo import NewsRepo
from app.adapters.db.user_repo import UserRepo
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.v1.news_router import router as news_router
from app.api.v1.user_router import router as user_router
from app.api.v1.rec_router import router as rec_router
from app.api.v1.rag_router import router as rag_router
from app.api.v1.forecast_router import router as forecast_router
from app.api.v1.stocks_router import router as stocks_router

from app.api.v1.auth_router import router as auth_router
from app.api.v1.user_router import router as user_router
from app.utils.healthy import check_database_connection


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 为你的模块设置更详细的日志级别
logging.getLogger("app.services.stock_recommender").setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context"""
    async with init_mongo_via_ssh(), init_postgres_sync():
        # 启动阶段
        user_repo = UserRepo()
        news_repo = NewsRepo()
        await user_repo.ensure_indexes()
        print("✅ MongoDB indexes ensured at startup.")
        # 交回控制权，开始处理请求
        yield
        # 关闭阶段（需要额外清理就放这里）
        print("🛑 App shutting down... (cleanup if needed)")


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
        base = LocalEmbeddingProvider(settings.ST_MODEL)  # e.g. 384
    elif provider in ("openai", "oai"):
        # 如果你已有 OpenAI 适配器，取消上面的 import 并启用这里
        # return OpenAIEmbedder(model=settings.OAI_EMBED_MODEL, api_key=settings.OPENAI_API_KEY)
        raise RuntimeError("EMBEDDING_PROVIDER=openai 尚未接入具体适配器，请实现 app/adapters/embeddings/openai_embedder.py 后再启用。")
    # 兜底
    else:
        base = HashingEmbedder(dim=max(64, settings.PROJECTION_DIM))

    # 投影总开关：只要 PROJECTION_METHOD 不是 none，就包一层
    method = (settings.PROJECTION_METHOD or "srp").lower().strip()
    if method != "none":
        emb = ProjectingEmbedder(
            base_embedder=base,
            method=method,
            proj_dim=settings.PROJECTION_DIM,
            seed=settings.PROJECTION_SEED
        )
        return emb
    return base

def _build_repos(embedder_dim: int):
    backend = (settings.VECTOR_BACKEND or "").lower().strip()

    if backend in ("pgvector", "pg", "postgres"):
        print("[RepoInit] ProfileRepo = PgProfileRepo, NewsRepo/EventRepo = Mongo")
        news_repo = MongoNewsRepo(settings.MONGO_URI, db_name=settings.MONGO_DB)
        ev_repo   = MongoEventRepo(settings.MONGO_URI, db_name=settings.MONGO_DB)
        prof_repo = PgProfileRepo(settings.PG_DSN, dim=embedder_dim)
        return news_repo, prof_repo, ev_repo

    if backend in ("mongo", "mongodb"):
        print("[RepoInit] ProfileRepo = MongoProfileRepo, NewsRepo/EventRepo = Mongo")
        news_repo = MongoNewsRepo(settings.MONGO_URI, db_name=settings.MONGO_DB)
        ev_repo   = MongoEventRepo(settings.MONGO_URI, db_name=settings.MONGO_DB)
        prof_repo = MongoProfileRepo(settings.MONGO_URI, db_name=settings.MONGO_DB, dim=embedder_dim)
        return news_repo, prof_repo, ev_repo
    else:
        print("[RepoInit] ProfileRepo = InMemoryProfileRepo, NewsRepo/EventRepo = InMemory")
        # 兜底：全部内存
        return InMemoryNewsRepo(), InMemoryProfileRepo(dim=embedder_dim), InMemoryEventRepo()

embedder = _build_embedder()
news_repo, prof_repo, ev_repo = _build_repos(
    embedder_dim=getattr(embedder, "dim", settings.DEFAULT_VECTOR_DIM)
)

svc = NewsService(news_repo, prof_repo, ev_repo, embedder)
sched = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sched

    # 可选：启动时导入种子数据
    try:
        svc.ingest([NewsItem(**n) for n in SEED_NEWS])
    except Exception:
        pass

    # —— 启动调度器（仅 dev/DEBUG）——
    # if settings.ENV.lower() == "dev" or settings.DEBUG:
    #     try:
    #         sched = create_scheduler(app, news_repo=news_repo, embedder=embedder)
    #         sched.start()
    #         print("[Scheduler] started with cron jobs")
    #     except Exception as e:
    #         print(f"[Scheduler] failed to start: {e}")

    # —— 启动调度器（仅当 ENV=dev/DEBUG 且 ENABLE_SCHEDULER=1）——
    if (settings.ENV.lower() == "dev" or settings.DEBUG) and getattr(settings, "ENABLE_SCHEDULER", 0):
        try:
            sched = create_scheduler(app, news_repo=news_repo, embedder=embedder)
            sched.start()
            print("[Scheduler] started with cron jobs")
        except Exception as e:
            print(f"[Scheduler] failed to start: {e}")
    else:
        print("[Scheduler] disabled (set ENABLE_SCHEDULER=1 to enable)")
    # 应用运行中
    yield

    # —— 关闭调度器 ——
    if sched is not None:
        try:
            sched.shutdown(wait=False)
            print("[Scheduler] shutdown ok")
        except Exception as e:
            print(f"[Scheduler] shutdown error: {e}")

def create_app() -> FastAPI:
    """
    构建 FastAPI 应用，加载路由、中间件、事件处理等
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description="Finsight Backend APIs",
        lifespan=lifespan,
    )

    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # 生产环境应该限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 中间件
    app.add_middleware(RequestContextMiddleware)

    # 全局异常处理
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # 注册路由
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(news_router)
    app.include_router(rec_router)
    app.include_router(rag_router)
    app.include_router(forecast_router)
    app.include_router(stocks_router)
    @app.get("/")
    async def root():
        """
        根路径
        """
        db_status = check_database_connection()
        return {
            "message": "股票推荐系统 API",
            "status": "running",
            "version": "1.0.0",
            "database_status": db_status,
            "endpoints": {
                "文档": "/docs",
                "健康检查": "/health",
                "股票数据": "/api/stocks",
                "用户管理": "/api/users"
            }
        }
    app.include_router(user_router)

    # 调试与维护路由（仅 dev/DEBUG）
    if settings.ENV.lower() == "dev" or settings.DEBUG:
        from app.api.v1.debug_router import router as debug_router
        app.include_router(debug_router)

    @app.get("/health")
    async def health_check():
        """
        健康检查端点
        """
        db_status = check_database_connection()
        return {
            "status": "healthy",
            "service": "stock_recommendation",
            "database": db_status,
            "timestamp": "2024-01-01T00:00:00Z"  # 实际应该用datetime
        }

    @app.get("/debug/status")
    async def debug_status():
        """
        调试状态检查
        """
        db_status = check_database_connection()
        return {
            "status": "running",
            "service": "stock_recommendation",
            "version": "1.0.0",
            "database": db_status,
        }

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # 开发模式下自动重载
    )