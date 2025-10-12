from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    AUTH_SECRET_KEY: str = Field(alias="AUTH_SECRET_KEY")
    AUTH_ALGORITHM: str = Field("HS256", alias="AUTH_ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(60 * 24, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    APP_NAME: str = "Finsight"
    MONGO_URI: str = Field(..., env="MONGO_URI")
    MONGO_DB: str = Field("finsight", env="MONGO_DB")
    REDIS_URL: str = Field(..., env="REDIS_URL")
    EMBEDDING_PROVIDER: str = Field("openai", env="EMBEDDING_PROVIDER")
    VECTOR_BACKEND: str = Field("mongo", env="VECTOR_BACKEND")  # mongo | pgvector
    LLM_PROVIDER: str = Field("openai", env="LLM_PROVIDER")
    NEWS_FETCH_CRON: str = "0 * * * *"  # 每小时
    DEFAULT_VECTOR_DIM: int = 32

    # LLM Setting
    LLM_PROVIDER: str = Field("deepseek_openai", alias="LLM_PROVIDER")
    LLM_OPENAI_BASE: str = Field("http://127.0.0.1:8000/v1", alias="LLM_OPENAI_BASE")
    LLM_OPENAI_API_KEY: str = Field("sk-local-placeholder", alias="LLM_OPENAI_API_KEY")
    LLM_MODEL: str = Field("deepseek-8b", alias="LLM_MODEL")

settings = Settings()