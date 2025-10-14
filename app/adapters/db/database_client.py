# app/adapters/db/database_client.py
from contextlib import asynccontextmanager
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from sshtunnel import SSHTunnelForwarder
import asyncpg
from app.config import settings

# ---- 全局句柄 ----
_tunnel: Optional[SSHTunnelForwarder] = None
_mongo_client: Optional[AsyncIOMotorClient] = None
_mongo_db: Optional[AsyncIOMotorDatabase] = None
_pg_db: Optional[asyncpg.Connection] = None


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
        remote_bind_addresses=[
            (settings.REMOTE_MONGO_HOST, settings.REMOTE_MONGO_PORT),  # idx 0
            (settings.REMOTE_PG_HOST, settings.REMOTE_PG_PORT),        # idx 1
        ],
        # ✅ 多端口时用 local_bind_addresses（复数）
        local_bind_addresses=[
            (settings.LOCAL_MONGO_HOST, 0),  # 本地自动分配端口
            (settings.LOCAL_PG_HOST, 0),
        ],
    )
    server.start()

    local_mongo_host = server.local_bind_hosts[0]
    local_mongo_port = server.local_bind_ports[0]
    local_pg_host = server.local_bind_hosts[1]
    local_pg_port = server.local_bind_ports[1]

    print("🔐 SSH tunnel established:")
    print(f"  → MongoDB: {local_mongo_host}:{local_mongo_port} "
          f"→ {settings.REMOTE_MONGO_HOST}:{settings.REMOTE_MONGO_PORT}")
    print(f"  → PostgreSQL: {local_pg_host}:{local_pg_port} "
          f"→ {settings.REMOTE_PG_HOST}:{settings.REMOTE_PG_PORT}")

    # 可选：把本地端口写回 settings（如果你后续想用）
    settings.LOCAL_MONGO_PORT = local_mongo_port
    settings.LOCAL_PG_PORT = local_pg_port

    return server


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
        # ✅ 多端口用 local_bind_ports
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


@asynccontextmanager
async def init_postgres_via_ssh():
    """
    初始化 PostgreSQL 连接；若开启 SSH 则复用/建立隧道
    """
    global _tunnel, _pg_db

    started_tunnel_here = False
    if settings.SSH_TUNNEL:
        if _tunnel is None:
            _tunnel = _start_ssh_tunnel()
            started_tunnel_here = True
        local_port = _tunnel.local_bind_ports[1]
        host = "127.0.0.1"     # ✅ 通过隧道，一定连本地
        port = local_port
    else:
        host = settings.LOCAL_PG_HOST or "127.0.0.1"
        port = settings.LOCAL_PG_PORT or 5432
        print(f"🔌 Direct PostgreSQL: {host}:{port}")

    try:
        _pg_db = await asyncpg.connect(
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DB,
            host=host,
            port=port,
        )
        print("✅ PostgreSQL connected successfully.")
        yield _pg_db
    finally:
        if _pg_db:
            await _pg_db.close()
            _pg_db = None
            print("🛑 PostgreSQL connection closed.")
        # 仅在本函数启动了隧道时才关闭
        if started_tunnel_here and _tunnel:
            _tunnel.stop()
            _tunnel = None
            print("🛑 SSH tunnel closed.")


# ---- Getters ----

def get_mongo_db():
    if _mongo_db is None:
        raise ConnectionError("MongoDB 连接不可用")
    return _mongo_db

def get_postgres_db():
    if _pg_db is None:
        raise ConnectionError("PostgreSQL 连接不可用")
    return _pg_db