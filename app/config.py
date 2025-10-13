from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    APP_NAME: str = "Finsight"
    MONGO_URI: str = Field(..., env="MONGO_URI")
    MONGO_DB_STOCK: str = Field("finsight", env="MONGO_DB_STOCK")
    POSTGRES_URI: str = Field(..., env="POSTGRES_URI")
    REDIS_URL: str = Field(..., env="REDIS_URL")
    EMBEDDING_PROVIDER: str = Field("openai", env="EMBEDDING_PROVIDER")
    VECTOR_BACKEND: str = Field("mongo", env="VECTOR_BACKEND")  # mongo | pgvector
    LLM_PROVIDER: str = Field("openai", env="LLM_PROVIDER")
    NEWS_FETCH_CRON: str = "0 * * * *"  # 每小时
    DEFAULT_VECTOR_DIM: int = 32

    class Config:
        env_file = ".env"

settings = Settings()