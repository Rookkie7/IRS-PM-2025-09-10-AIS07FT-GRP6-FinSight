from __future__ import annotations

from config import settings
from adapters.db.news_repo_mongo import NewsRepoMongo
from adapters.vector.mongo_vector_index import MongoVectorIndex
from adapters.embeddings.sentence_transformers_embed import LocalEmbeddingProvider
from services.news_service import NewsService
from services.rec_service import RecService
from services.rag_service import RagService
from app.services.forecast_service import ForecastService, PriceProviderPort, ForecastConfig
from app.forecasters.arima_forecaster import ArimaForecaster
# from app.forecasters.prophet_forecaster import ProphetForecaster
# from app.forecasters.lgbm_forecaster import LgbmForecaster
# from app.forecasters.lstm_forecaster import LstmForecaster

from typing import List
import math
def get_vector_index():
    # 可根据 settings.VECTOR_BACKEND 切换到 pgvector_index.PgVectorIndex()
    return MongoVectorIndex(collection_name="news")  # 你也可以为 stocks 建独立索引/集合

def get_embedder():
    return LocalEmbeddingProvider()  # 或 OpenAIEmbeddingProvider()

def get_news_service():
    return NewsService(
        repo=NewsRepoMongo(),
        embedder=get_embedder(),
        index=get_vector_index(),
        dim=settings.DEFAULT_VECTOR_DIM
    )

def get_rec_service():
    return RecService(vector_index=get_vector_index(), dim=settings.DEFAULT_VECTOR_DIM)

def get_rag_service():
    return RagService(index=get_vector_index(), llm=None, dim=settings.DEFAULT_VECTOR_DIM)



# ========= FORECAST SERVICE ========================
class CsvPriceProvider(PriceProviderPort):
    async def get_recent_closes(self, ticker: str, lookback_days: int = 252):
        # TODO: 替换为你项目真实的数据通道 (Mongo/CSV/yfinance/…)
        # import pandas as pd
        # import pathlib
        # path = pathlib.Path("data")/f"{ticker}.csv"  # 假设有 data/AAPL.csv
        # df = pd.read_csv(path)  # 必须包含按日期升序的 Close 列
        # closes = df["Close"].astype(float).tail(lookback_days).tolist()
        # return closes
        return [100 + i * 0.12 for i in range(260)]

class MockPriceProvider(PriceProviderPort):
    """示例：生成一段轻微上升+小噪声的价格序列。请替换成你的真实数据源。"""
    async def get_recent_closes(self, ticker: str, lookback_days: int = 252) -> List[float]:
        base = 100.0
        series: List[float] = []
        drift = 0.0008
        vol = 0.01
        import random
        for i in range(lookback_days + 10):
            shock = random.gauss(0, vol)
            base = max(1.0, base * (1.0 + drift + shock))
            series.append(base)
        return series

def get_forecast_service() -> ForecastService:
    provider = MockPriceProvider()
    provider = CsvPriceProvider()
    cfg = ForecastConfig(lookback_days=252, ma_window=20)

    forecasters = {
        "arima": ArimaForecaster(order=(1, 1, 1)),
        # "prophet": ProphetForecaster(),
        # "lgbm": LgbmForecaster(booster=loaded_booster),
        # "lstm": LstmForecaster(model=loaded_torch_model),
    }

    return ForecastService(price_provider=provider, cfg=cfg, forecasters=forecasters)