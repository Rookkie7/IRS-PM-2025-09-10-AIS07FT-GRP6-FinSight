# app/api/v1/macro_router.py
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
import os
from app.services.macro_service import MacroService

router = APIRouter(prefix="/macro", tags=["macro"])

def get_macro_service() -> MacroService:
    provider = os.getenv("MARKET_PROVIDER", "yahoo").lower()
    # 可通过环境变量 MARKETS_TTL_SEC 覆盖缓存秒数
    ttl_sec = int(os.getenv("MARKETS_TTL_SEC", "30"))
    return MacroService(provider="alpha" if provider == "alpha" else "yahoo", ttl_sec=ttl_sec)

@router.get("/spx")
async def get_spx(svc: MacroService = Depends(get_macro_service)):
    try:
        return await svc.get_spx()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vix")
async def get_vix(svc: MacroService = Depends(get_macro_service)):
    try:
        return await svc.get_vix()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment")
async def get_sentiment(svc: MacroService = Depends(get_macro_service)):
    try:
        return await svc.get_sentiment()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))