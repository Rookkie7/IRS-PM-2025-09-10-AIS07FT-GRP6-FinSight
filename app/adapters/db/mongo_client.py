from __future__ import annotations
import os


from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from sshtunnel import SSHTunnelForwarder
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from sshtunnel import SSHTunnelForwarder
from app.config import settings   # ✅ 用 settings 读取 .env

_tunnel = None
_client = None
_db = None

def _start_ssh_tunnel() -> SSHTunnelForwarder:
    server = SSHTunnelForwarder(
        ssh_address_or_host=(settings.SSH_HOST, settings.SSH_PORT),
        ssh_username=settings.SSH_USER,
        ssh_pkey=settings.SSH_PEM_PATH,
        remote_bind_address=(settings.REMOTE_MONGO_HOST, settings.REMOTE_MONGO_PORT),
        local_bind_address=(settings.LOCAL_BIND_HOST, settings.LOCAL_BIND_PORT),
    )
    server.start()
    print(f"🔐 SSH tunnel: {settings.LOCAL_BIND_HOST}:{server.local_bind_port} → "
          f"{settings.REMOTE_MONGO_HOST}:{settings.REMOTE_MONGO_PORT}")
    return server


def _build_mongo_uri(port: int) -> str:
    # 隧道建立后，客户端连接到本地端口
    return f"mongodb://127.0.0.1:{port}"

@asynccontextmanager
async def init_mongo_via_ssh():
    global _tunnel, _client, _db
    if settings.SSH_TUNNEL:
        _tunnel = _start_ssh_tunnel()
        uri = f"mongodb://127.0.0.1:{_tunnel.local_bind_port}"
    else:
        uri = settings.MONGO_URI or "mongodb://127.0.0.1:27017"
        print(f"🔌 Direct Mongo URI: {uri}")

    _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    _db = _client[settings.MONGO_DB]
    try:
        await _db.command("ping")
        yield
    finally:
        if _client: _client.close()
        if _tunnel: _tunnel.stop()
        _client = _db = _tunnel = None


def get_db() -> AsyncIOMotorDatabase:
    assert _db is not None, "Mongo 未初始化，请在应用启动时调用 init_mongo_via_ssh()"
    return _db