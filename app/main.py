# app/main.py
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.adapters.db.database_client import init_mongo_via_ssh, init_postgres_sync
from app.adapters.db.news_repo import NewsRepo
from app.adapters.db.user_repo import UserRepo
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.v1.news_router import router as news_router
from app.api.v1.rec_router import router as rec_router
from app.api.v1.rag_router import router as rag_router
from app.api.v1.forecast_router import router as forecast_router
from app.api.v1.stocks_router import router as stocks_router

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

    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # 生产环境应该限制具体域名
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