from app.adapters.db.price_provider_mongo import MongoStockPriceProvider
from app.adapters.db.user_repo import UserRepo
from sqlalchemy.orm.session import Session
from fastapi import Depends
from pymongo.database import Database
from app.adapters.llm.openai_llm import OpenAICompatLLM
from app.services.auth_service import AuthService
from app.services.stock_recommender import MultiObjectiveRecommender
from app.services.stock_service import StockService
from app.services.user_service import UserService
from fastapi import HTTPException

from config import settings
from adapters.db.news_repo import NewsRepo
from adapters.vector.mongo_vector_index import MongoVectorIndex
from adapters.embeddings.sentence_transformers_embed import LocalEmbeddingProvider
from services.news_service import NewsService
from services.rec_service import RecService
from services.rag_service import RagService
from services.forecast_service import ForecastService, ForecastConfig
from adapters.db.database_client import get_mongo_db, get_postgres_session


def get_user_repo() -> UserRepo:
    return UserRepo()

def get_auth_service():
    return AuthService(repo=get_user_repo())

def get_user_service(
        db: Session = Depends(get_postgres_session),):
    return UserService(db=db, repo=get_user_repo(), dim=20)

def get_stock_service(
        postgres_db: Session = Depends(get_postgres_session),
        mongo_db: Database = Depends(get_mongo_db),
):
    return StockService(
        postgres_db=postgres_db,
        mongo_db=mongo_db,
    )

def get_multi_objective_recommender(
    postgres_db: Session = Depends(get_postgres_session),
    mongo_db: Database = Depends(get_mongo_db)
):
    return MultiObjectiveRecommender(postgres_db, mongo_db)

def get_llm():
    if settings.LLM_PROVIDER == "deepseek_openai":
        return OpenAICompatLLM(
            base_url=settings.LLM_OPENAI_BASE,
            api_key=settings.LLM_OPENAI_API_KEY,
            model=settings.LLM_MODEL,
        )
        # fallback
    return OpenAICompatLLM()

def get_query_embedder():
    # You can also use OpenAI Embeddings; here we use the local SB model
    return LocalEmbeddingProvider()

def get_vector_index():
    # Can be switched to pgvector_index.PgVectorIndex() according to settings.VECTOR_BACKEND
    return MongoVectorIndex(collection_name="news")  # 你也可以为 stocks 建独立索引/集合

def get_embedder():
    return LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")  # 或 OpenAIEmbeddingProvider()

def get_news_index():
    return MongoVectorIndex(collection_name="news")

def get_news_service():
    return NewsService(
        repo=NewsRepo(),
        embedder=get_embedder(),
        index=get_vector_index(),
        dim=settings.DEFAULT_VECTOR_DIM
    )

def get_rec_service():
    return RecService(vector_index=get_vector_index(), dim=settings.DEFAULT_VECTOR_DIM)

def get_rag_service():
    if not settings.ragflow_enabled:
        # 未配置就禁用或返回一个空实现
        # 选1：直接 503 更清晰
        raise HTTPException(status_code=503, detail="RAG service is not configured.")
    return RagService()


def get_price_provider():
    return MongoStockPriceProvider(collection_name="stocks")

def get_forecast_service(provider = Depends(get_price_provider)):
    cfg = ForecastConfig(lookback_days=252, ma_window=20)
    return ForecastService(price_provider=provider, cfg=cfg)