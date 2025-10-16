# app/main.py
from __future__ import annotations
from contextlib import asynccontextmanager

import re
from typing import Optional
import uvicorn
from fastapi import FastAPI

from app.adapters.db.database_client import init_mongo_via_ssh, init_postgres_sync
from app.adapters.db.news_repo import NewsRepo
from app.adapters.db.user_repo import UserRepo
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.adapters.db.database_client import init_mongo_via_ssh, init_postgres_via_ssh
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
from app.repositories.mongo_repos import MongoNewsRepo, MongoEventRepo, MongoProfileRepo
from app.repositories.pg_profile_repo import PgProfileRepo
from app.repositories.inmemory import InMemoryNewsRepo, InMemoryProfileRepo, InMemoryEventRepo
from app.services.news_service import NewsService

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

from app.api.v1.auth_router import router as auth_router
from app.api.v1.user_router import router as user_router
from app.utils.healthy import check_database_connection


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

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产建议收紧
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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