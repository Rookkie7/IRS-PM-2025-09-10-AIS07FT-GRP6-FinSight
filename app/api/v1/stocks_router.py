from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pymongo.database import Database
from typing import List

from app.adapters.db.database_client import get_postgres_db, get_mongo_db
from app.services.stock_service import StockService

router = APIRouter(prefix="/api/stocks", tags=["stocks"])


@router.post("/fetch-raw-data")
async def fetch_raw_stock_data(
        symbols: str = Query(..., description="股票代码列表，例如：AAPL,GOOG,MSFT"),
        postgres_db: Session = Depends(get_postgres_db),
        mongo_db: Database = Depends(get_mongo_db)
):
    """
    从yfinance获取原始股票数据并存入MongoDB
    """
    try:
        stock_service = StockService(postgres_db, mongo_db)

        # 解析逗号分隔的字符串
        symbols_to_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

        result = stock_service.fetch_stock_data(symbols_to_list)

        return {
            "ok": True,
            "message": f"成功获取 {len(result)} 只股票的原始数据",
            "fetched_symbols": list(result.keys()),
            "count": len(result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取股票数据失败: {str(e)}")


@router.post("/update-vectors")
async def update_stock_vectors(
        symbols: str = Query(..., description="股票代码列表，逗号分隔"),  # 只接受字符串
        postgres_db: Session = Depends(get_postgres_db),
        mongo_db: Database = Depends(get_mongo_db)
):
    """
    更新股票向量（从MongoDB计算后存入PostgreSQL）
    """
    try:
        stock_service = StockService(postgres_db, mongo_db)

        # 解析逗号分隔的字符串
        symbols_to_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

        if not symbols_to_list:
            raise HTTPException(status_code=400, detail="股票代码列表为空")

        result = stock_service.update_stock_vectors(symbols_to_list)

        return {
            "ok": True,
            "message": f"成功更新 {result['updated']} 只股票向量",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新股票向量失败: {str(e)}")


@router.get("/recommend")
async def get_stock_recommendations(
        user_id: str = Query(..., description="用户ID"),
        top_k: int = Query(5, ge=1, le=20, description="推荐数量"),
        postgres_db: Session = Depends(get_postgres_db),
        mongo_db: Database = Depends(get_mongo_db)
):
    """
    获取个性化股票推荐（基于20维向量）
    """
    from app.services.user_service import UserService

    try:
        user_service = UserService(postgres_db)
        stock_service = StockService(postgres_db, mongo_db)

        # 检查用户是否存在
        user_profile = user_service.get_user_profile(user_id)
        if not user_profile:
            raise HTTPException(
                status_code=404,
                detail=f"用户 {user_id} 的画像不存在，请先初始化用户画像"
            )

        # 获取用户20维向量
        user_vector = user_profile.get_profile_vector_20d()

        # 获取推荐
        recommendations = stock_service.recommend_stocks(user_vector, top_k)

        return {
            "ok": True,
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations),
            "vector_dimension": len(user_vector)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取推荐失败: {str(e)}")


@router.get("/list")
async def get_all_stocks(
        postgres_db: Session = Depends(get_postgres_db),
        mongo_db: Database = Depends(get_mongo_db)
):
    """
    获取所有已存储的股票列表
    """
    try:
        stock_service = StockService(postgres_db, mongo_db)
        stocks = stock_service.get_all_stocks()

        return {
            "ok": True,
            "stocks": stocks,
            "count": len(stocks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取股票列表失败: {str(e)}")


@router.get("/raw-data/{symbol}")
async def get_stock_raw_data(
        symbol: str,
        mongo_db: Database = Depends(get_mongo_db)
):
    """
    获取股票的原始数据（从MongoDB）
    """
    try:
        collection = mongo_db["stock_raw_data"]
        raw_data = collection.find_one({"symbol": symbol.upper()})

        if not raw_data:
            raise HTTPException(status_code=404, detail=f"股票 {symbol} 的原始数据未找到")

        # 移除MongoDB的_id字段
        raw_data.pop('_id', None)

        return {
            "ok": True,
            "symbol": symbol,
            "raw_data": raw_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取原始数据失败: {str(e)}")