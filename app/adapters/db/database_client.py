from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

# 数据库配置
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://stock_user:stock_password@localhost:5432/stock_recommendation"
)

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB", "stock_data")

# PostgreSQL配置
try:
    postgres_engine = create_engine(
        POSTGRES_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=postgres_engine)

    # 测试连接
    with postgres_engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("✅ PostgreSQL连接成功")

except Exception as e:
    logger.error(f"❌ PostgreSQL连接失败: {e}")
    raise

# MongoDB配置
try:
    mongo_client = MongoClient(
        MONGO_URL,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=10000,
        socketTimeoutMS=10000
    )
    # 测试连接
    mongo_client.admin.command('ping')
    mongo_db = mongo_client[MONGO_DB_NAME]
    logger.info("✅ MongoDB连接成功")
except ConnectionFailure as e:
    logger.error(f"❌ MongoDB连接失败: {e}")
    mongo_client = None
    mongo_db = None


def get_postgres_db():
    """获取PostgreSQL数据库会话"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logger.error(f"数据库会话错误: {e}")
        raise
    finally:
        db.close()


def get_mongo_db():
    """获取MongoDB数据库"""
    if mongo_db is None:
        raise ConnectionError("MongoDB连接不可用")
    return mongo_db


def create_tables():
    """创建PostgreSQL表"""
    try:
        from app.model.models import Base
        Base.metadata.create_all(bind=postgres_engine)
        logger.info("✅ PostgreSQL表创建成功")
    except Exception as e:
        logger.error(f"❌ 创建表失败: {e}")
        raise


def check_database_connection():
    """检查数据库连接状态"""
    status = {
        "postgresql": False,
        "mongodb": False
    }

    # 检查PostgreSQL
    try:
        with postgres_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["postgresql"] = True
        logger.info("✅ PostgreSQL连接检查成功")
    except Exception as e:
        logger.error(f"PostgreSQL连接检查失败: {e}")
        status["postgresql"] = False

    # 检查MongoDB
    try:
        if mongo_client:
            mongo_client.admin.command('ping')
            status["mongodb"] = True
            logger.info("✅ MongoDB连接检查成功")
        else:
            status["mongodb"] = False
    except Exception as e:
        logger.error(f"MongoDB连接检查失败: {e}")
        status["mongodb"] = False

    return status