# app/adapters/db/database_client.py
from contextlib import asynccontextmanager
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sshtunnel import SSHTunnelForwarder

from app.config import settings

# ---- 全局句柄 ----
_tunnel: Optional[SSHTunnelForwarder] = None

# Mongo
_mongo_client: Optional[AsyncIOMotorClient] = None
_mongo_db: Optional[AsyncIOMotorDatabase] = None

# Postgres（同步）
_pg_engine = None
_SessionLocal: Optional[sessionmaker] = None


def _start_ssh_tunnel() -> SSHTunnelForwarder:
    """
    启动 SSH 隧道，同时转发 MongoDB (27017) 和 PostgreSQL (5432)
    """
    assert settings.SSH_HOST and settings.SSH_USER and settings.SSH_PEM_PATH, \
        "SSH_HOST/SSH_USER/SSH_PEM_PATH 必须配置"

    server = SSHTunnelForwarder(
        ssh_address_or_host=(settings.SSH_HOST, settings.SSH_PORT),
        ssh_username=settings.SSH_USER,
        ssh_pkey=settings.SSH_PEM_PATH,
        remote_bind_addresses=[  # 索引 0: Mongo, 1: Postgres
            (settings.REMOTE_MONGO_HOST, settings.REMOTE_MONGO_PORT),
            (settings.REMOTE_PG_HOST, settings.REMOTE_PG_PORT),
        ],
        local_bind_addresses=[
            (settings.LOCAL_MONGO_HOST, 0),  # 本地自动分配端口
            (settings.LOCAL_PG_HOST, 0),
        ],
    )
    server.start()

    mongo_host, mongo_port = server.local_bind_hosts[0], server.local_bind_ports[0]
    pg_host, pg_port       = server.local_bind_hosts[1], server.local_bind_ports[1]

    print("🔐 SSH tunnel established:")
    print(f"  → MongoDB:   {mongo_host}:{mongo_port} → {settings.REMOTE_MONGO_HOST}:{settings.REMOTE_MONGO_PORT}")
    print(f"  → PostgreSQL:{pg_host}:{pg_port}     → {settings.REMOTE_PG_HOST}:{settings.REMOTE_PG_PORT}")

    # 可选：把端口写回 settings（后续其他地方想用也方便）
    settings.LOCAL_MONGO_PORT = mongo_port
    settings.LOCAL_PG_PORT = pg_port
    return server


# ------------------ Mongo (异步 motor) ------------------

@asynccontextmanager
async def init_mongo_via_ssh():
    """
    初始化 Mongo 连接；若开启 SSH 则复用/建立隧道
    """
    global _tunnel, _mongo_client, _mongo_db

    started_tunnel_here = False
    if settings.SSH_TUNNEL:
        if _tunnel is None:
            _tunnel = _start_ssh_tunnel()
            started_tunnel_here = True
        local_port = _tunnel.local_bind_ports[0]
        uri = f"mongodb://127.0.0.1:{local_port}"
    else:
        uri = settings.MONGO_URI or "mongodb://127.0.0.1:27017"
        print(f"🔌 Direct Mongo URI: {uri}")

    _mongo_client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    _mongo_db = _mongo_client[settings.MONGO_DB]
    try:
        await _mongo_db.command("ping")
        yield
    finally:
        if _mongo_client:
            _mongo_client.close()
            _mongo_client = None
            _mongo_db = None
        # 仅在本函数启动了隧道时才关闭（避免影响 PG）
        if started_tunnel_here and _tunnel:
            _tunnel.stop()
            _tunnel = None


def get_mongo_db() -> AsyncIOMotorDatabase:
    if _mongo_db is None:
        raise ConnectionError("MongoDB 连接不可用（lifespan 未初始化？）")
    return _mongo_db


# ------------------ Postgres（同步 SQLAlchemy） ------------------

def _build_sync_pg_url(host: str, port: int) -> str:
    return (
        f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{host}:{port}/{settings.POSTGRES_DB}"
    )

def _init_postgres_sync():
    """创建同步 SQLAlchemy Engine 与 Session 工厂"""
    global _pg_engine, _SessionLocal, _tunnel

    if settings.SSH_TUNNEL:
        if _tunnel is None:
            # 如果 Mongo 还没开隧道，这里会新建并共享
            _tunnel = _start_ssh_tunnel()
        host, port = "127.0.0.1", _tunnel.local_bind_ports[1]
    else:
        host = settings.LOCAL_PG_HOST or "127.0.0.1"
        port = settings.LOCAL_PG_PORT or 5432
        print(f"🔌 Direct PostgreSQL: {host}:{port}")

    url = _build_sync_pg_url(host, port)
    _pg_engine = create_engine(url, pool_pre_ping=True, future=True)
    _SessionLocal = sessionmaker(bind=_pg_engine, autocommit=False, autoflush=False, class_=Session)
    print("✅ PostgreSQL sync engine initialized.")

def _dispose_postgres_sync():
    global _pg_engine, _SessionLocal
    if _pg_engine is not None:
        _pg_engine.dispose()
        _pg_engine = None
    _SessionLocal = None
    print("🛑 PostgreSQL engine disposed.")

@asynccontextmanager
async def init_postgres_sync():
    """
    lifespan 里调用：初始化同步 SQLAlchemy（即使外层是 async 也没关系）
    """
    _init_postgres_sync()
    try:
        yield
    finally:
        _dispose_postgres_sync()
        # 注意：不在这里停 SSH 隧道；由最后一个使用者关闭或统一在应用退出处关闭
        # 如果你只在一个地方创建了 _tunnel，可在主 lifespan 退出时统一 _tunnel.stop()


def get_postgres_session():
    """FastAPI 依赖：提供同步 Session（线程池执行）"""
    if _SessionLocal is None:
        raise RuntimeError("PostgreSQL session factory 未初始化（请在 lifespan 里先调用 init_postgres_sync()）")
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()