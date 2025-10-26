# app/adapters/db/database_client.py
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Optional, Generator

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from fastapi import Request
from starlette.datastructures import State
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sshtunnel import SSHTunnelForwarder

from app.config import settings

# ---- 全局句柄（兜底用；首选从 app.state 读）----
_tunnel: Optional[SSHTunnelForwarder] = None

# Mongo
_mongo_client: Optional[AsyncIOMotorClient] = None
_mongo_db: Optional[AsyncIOMotorDatabase] = None

# Postgres（同步 SQLAlchemy）
_pg_engine = None
_SessionLocal: Optional[sessionmaker] = None


def _start_ssh_tunnel() -> SSHTunnelForwarder:
    assert settings.SSH_HOST and settings.SSH_USER and settings.SSH_PEM_PATH, \
        "SSH_HOST/SSH_USER/SSH_PEM_PATH 必须配置"

    server = SSHTunnelForwarder(
        ssh_address_or_host=(settings.SSH_HOST, settings.SSH_PORT),
        ssh_username=settings.SSH_USER,
        ssh_pkey=settings.SSH_PEM_PATH,
        remote_bind_addresses=[
            (settings.REMOTE_MONGO_HOST, settings.REMOTE_MONGO_PORT),
            (settings.REMOTE_PG_HOST, settings.REMOTE_PG_PORT),
        ],
        local_bind_addresses=[
            (settings.LOCAL_MONGO_HOST, 0),
            (settings.LOCAL_PG_HOST, 0),
        ],
    )
    server.start()

    mongo_host, mongo_port = server.local_bind_hosts[0], server.local_bind_ports[0]
    pg_host, pg_port       = server.local_bind_hosts[1], server.local_bind_ports[1]

    print("🔐 SSH tunnel established:")
    print(f"  → MongoDB:   {mongo_host}:{mongo_port} → {settings.REMOTE_MONGO_HOST}:{settings.REMOTE_MONGO_PORT}")
    print(f"  → PostgreSQL:{pg_host}:{pg_port}     → {settings.REMOTE_PG_HOST}:{settings.REMOTE_PG_PORT}")

    # 写回 settings，便于其它地方取到映射端口
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
    """创建同步 SQLAlchemy Engine 与 Session 工厂（模块级兜底副本）"""
    global _pg_engine, _SessionLocal, _tunnel

    if settings.SSH_TUNNEL:
        if _tunnel is None:
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
    return url  # 便于调试

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
    ✅ lifespan 里调用：
       async with init_postgres_sync() as pg:
           app.state.pg_session_factory = pg["SessionLocal"]
           app.state.pg_engine = pg["engine"]
           app.state.pg_pool = pg["pool"]
    """
    dsn = _init_postgres_sync()
    try:
        yield {
            "engine": _pg_engine,
            "SessionLocal": _SessionLocal,
            "dsn": dsn,
        }
    finally:
        _dispose_postgres_sync()
        # 隧道不要在这里停，由最外层统一关闭（若与 Mongo 复用）

# ---------- 依赖注入：从 app.state 读取；读不到回退模块全局 ----------
def _ensure_state(obj) -> State:
    # 保险：部分测试环境/脚本对象可能没有 state
    if not hasattr(obj, "state"):
        obj.state = State()
    return obj.state

def get_postgres_session(request: Request):
    SessionLocal = getattr(getattr(request.app, "state", object()), "pg_session_factory", None)
    if SessionLocal is None:
        # 回退模块级（如果有人没用 lifespan 也能工作）
        global _SessionLocal
        SessionLocal = _SessionLocal

    if SessionLocal is None:
        raise RuntimeError("PostgreSQL session factory 未初始化（请在 lifespan 里先调用 init_postgres_sync()）")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()