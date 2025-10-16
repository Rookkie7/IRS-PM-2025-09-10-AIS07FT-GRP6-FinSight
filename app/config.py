# from pydantic_settings import BaseSettings
# from pydantic import Field
#
# class Settings(BaseSettings):
#     APP_NAME: str = "Finsight"
#     MONGO_URI: str = Field(..., env="MONGO_URI")
#     MONGO_DB: str = Field("finsight", env="MONGO_DB")
#     REDIS_URL: str = Field(..., env="REDIS_URL")
#     EMBEDDING_PROVIDER: str = Field("openai", env="EMBEDDING_PROVIDER")
#     VECTOR_BACKEND: str = Field("mongo", env="VECTOR_BACKEND")  # mongo | pgvector
#     LLM_PROVIDER: str = Field("openai", env="LLM_PROVIDER")
#     NEWS_FETCH_CRON: str = "0 * * * *"  # 每小时
#     DEFAULT_VECTOR_DIM: int = 32
#
#     class Config:
#         env_file = ".env"
#
# settings = Settings()


# app/config.py
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# 项目根目录（FinSight_BackEnd），无论从哪里运行都能定位到 .env
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

class Settings(BaseSettings):
    APP_NAME: str = "Finsight"

    # 给出开发期默认值（本地先跑通，不会因为没配 .env 就挂）
    MONGO_URI: str = Field("mongodb://localhost:27017/finsight")
    MONGO_DB: str = Field("finsight")
    REDIS_URL: str = Field("redis://localhost:6379/0")

    EMBEDDING_PROVIDER: str = Field("openai")
    VECTOR_BACKEND: str = Field("mongo")  # mongo | pgvector
    LLM_PROVIDER: str = Field("openai")
    NEWS_FETCH_CRON: str = "0 * * * *"
    DEFAULT_VECTOR_DIM: int = 32

    # pydantic-settings v2 写法：显式指定 .env 的绝对路径
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

settings = Settings()
