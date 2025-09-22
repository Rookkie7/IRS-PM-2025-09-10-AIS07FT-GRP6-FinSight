from config import settings
from adapters.db.news_repo_mongo import NewsRepoMongo
from adapters.vector.mongo_vector_index import MongoVectorIndex
from adapters.embeddings.sentence_transformers_embed import LocalEmbeddingProvider
from services.news_service import NewsService
from services.rec_service import RecService
from services.rag_service import RagService
from services.forecast_service import ForecastService

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

def get_forecast_service():
    return ForecastService()