# app/main.py
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.adapters.db.mongo_client import init_mongo_via_ssh
from app.adapters.db.user_repo_mongo import UserRepoMongo
from app.config import settings
from app.api.v1.news_router import router as news_router
from app.api.v1.rec_router import router as rec_router
from app.api.v1.rag_router import router as rag_router
from app.api.v1.forecast_router import router as forecast_router
from app.api.v1.auth_router import router as auth_router
from app.api.v1.user_router import router as user_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context"""
    async with init_mongo_via_ssh():
        # 启动阶段
        repo = UserRepoMongo()
        await repo.ensure_indexes()
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

    # 注册路由
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(news_router)
    app.include_router(rec_router)
    app.include_router(rag_router)
    app.include_router(forecast_router)

    # 健康检查接口
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # 开发模式下自动重载
    )