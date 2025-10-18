from sqlalchemy.orm.session import Session
from fastapi import Depends
from pymongo.database import Database
from app.adapters.db.database_client import get_postgres_session,get_mongo_db
from app.adapters.db.user_repo import UserRepo
from app.adapters.llm.openai_llm import OpenAICompatLLM
from app.ports.storage import UserRepoPort
from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.services.stock_service import StockService
# from config import settings
from app.adapters.db.news_repo import NewsRepo
from app.adapters.vector.mongo_vector_index import MongoVectorIndex
from app.adapters.embeddings.sentence_transformers_embed import LocalEmbeddingProvider
from app.services.news_service import NewsService
from app.services.rec_service import RecService
from app.services.rag_service import RagService
from app.services.forecast_service import ForecastService
from app.config import settings

def get_query_embedder():
    # You can also use OpenAI Embeddings; here we use the local SB model
    return LocalEmbeddingProvider()

def get_vector_index():
    # Can be switched to pgvector_index.PgVectorIndex() according to settings.VECTOR_BACKEND
    return MongoVectorIndex(collection_name="news")  # 你也可以为 stocks 建独立索引/集合

def get_embedder():
    return LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")  # 或 OpenAIEmbeddingProvider()

def get_user_repo() -> UserRepo:
    return UserRepo()

def get_auth_service():
    return AuthService(repo=get_user_repo())

def get_user_service(
        db: Session = Depends(get_postgres_session),
        embedder = Depends(get_embedder),):
    return UserService(db=db, repo=get_user_repo(), embedder=embedder, dim=32)

def get_stock_service(
        postgres_db: Session = Depends(get_postgres_session),
        mongo_db: Database = Depends(get_mongo_db),
):
    return StockService(
        postgres_db=postgres_db,
        mongo_db=mongo_db,
    )

def get_llm():
    if settings.LLM_PROVIDER == "deepseek_openai":
        return OpenAICompatLLM(
            base_url=settings.LLM_OPENAI_BASE,
            api_key=settings.LLM_OPENAI_API_KEY,
            model=settings.LLM_MODEL,
        )
        # fallback
    return OpenAICompatLLM()


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
    return RagService(
        index=get_news_index(),
        news_repo=NewsRepo(),
        query_embedder=get_query_embedder(),
        llm=get_llm(),
        dim=32,
    )

def get_forecast_service():
    return ForecastService()