# app/main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv

from app.config import settings
from app.api.v1.stocks_router import router as stocks_router
from app.api.v1.users_router import router as users_router

from app.adapters.db.database_client import create_tables, check_database_connection


def create_app() -> FastAPI:
    """
    构建 FastAPI 应用，加载路由、中间件、事件处理等
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description="Finsight Backend APIs",
    )

    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应该限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(stocks_router)
    app.include_router(users_router)

    # 加载环境变量
    load_dotenv()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 创建数据库表
    try:
        create_tables()
        logger.info("✅ 数据库表创建成功")
    except Exception as e:
        logger.error(f"❌ 数据库表创建失败: {e}")
        raise

    # 检查数据库连接
    db_status = check_database_connection()
    logger.info(f"📊 数据库连接状态: {db_status}")

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