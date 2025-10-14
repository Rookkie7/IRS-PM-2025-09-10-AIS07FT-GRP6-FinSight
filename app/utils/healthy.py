# app/utils/health.py
from __future__ import annotations
import time
from typing import Dict, Any
from app.adapters.db.database_client import get_postgres_db, get_mongo_db  # 推荐：显式 getter


async def check_database_connection() -> Dict[str, Any]:
    """
    并不创建连接，只是**验证当前已初始化的连接**是否可用。
    - Mongo: db.command("ping")
    - Postgres: SELECT 1
    返回: {"mongo": {...}, "postgres": {...}, "ok": bool}
    """
    results: Dict[str, Any] = {
        "mongo": {"ok": False, "latency_ms": None, "error": None},
        "postgres": {"ok": False, "latency_ms": None, "error": None},
        "ok": False,
    }

    # 1) MongoDB
    try:
        db = get_mongo_db()  # 若未在 lifespan 初始化会抛错
        t0 = time.monotonic()
        await db.command("ping")
        t1 = time.monotonic()
        results["mongo"]["ok"] = True
        results["mongo"]["latency_ms"] = round((t1 - t0) * 1000, 2)
    except Exception as e:
        results["mongo"]["error"] = f"{type(e).__name__}: {e}"

    # 2) PostgreSQL
    try:
        pg_db = await get_postgres_db()
        if pg_db is None:
            raise RuntimeError("PostgreSQL connection is not initialized yet")
        t0 = time.monotonic()
        val = await pg_db.fetchval("SELECT 1;")
        t1 = time.monotonic()
        ok = (val == 1)
        results["postgres"]["ok"] = bool(ok)
        results["postgres"]["latency_ms"] = round((t1 - t0) * 1000, 2)
        if not ok:
            results["postgres"]["error"] = f"Unexpected SELECT 1 result: {val}"
    except Exception as e:
        results["postgres"]["error"] = f"{type(e).__name__}: {e}"

    # 汇总
    results["ok"] = results["mongo"]["ok"] and results["postgres"]["ok"]
    return results