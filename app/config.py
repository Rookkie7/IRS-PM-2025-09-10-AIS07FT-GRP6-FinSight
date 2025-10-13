from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    AUTH_SECRET_KEY: str = Field(alias="AUTH_SECRET_KEY")
    AUTH_ALGORITHM: str = Field("HS256", alias="AUTH_ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(60 * 24, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    APP_NAME: str = "Finsight"
    # SSH 隧道相关
    SSH_TUNNEL: bool = Field(False, alias="SSH_TUNNEL")
    SSH_HOST: str | None = None
    SSH_PORT: int = 22
    SSH_USER: str | None = None
    SSH_PEM_PATH: str | None = None
    REMOTE_MONGO_HOST: str = "127.0.0.1"
    REMOTE_MONGO_PORT: int = 27017
    LOCAL_BIND_HOST: str = "127.0.0.1"
    LOCAL_BIND_PORT: int = 0  # 0=随机

    # Mongo 连接串：当 SSH_TUNNEL=False 时需要；开启隧道时可不填
    MONGO_URI: str | None = Field(default=None, alias="MONGO_URI")
    MONGO_DB: str = "finsight"

    REDIS_URL: str = Field(..., env="REDIS_URL")
    EMBEDDING_PROVIDER: str = Field("openai", env="EMBEDDING_PROVIDER")
    VECTOR_BACKEND: str = Field("pgvector", env="VECTOR_BACKEND")  # mongo | pgvector
    NEWS_FETCH_CRON: str = "0 * * * *"  # 每小时
    DEFAULT_VECTOR_DIM: int = 32

    # LLM Setting
    LLM_PROVIDER: str = Field("deepseek_openai", alias="LLM_PROVIDER")
    LLM_OPENAI_BASE: str = Field("http://127.0.0.1:8000/v1", alias="LLM_OPENAI_BASE")
    LLM_OPENAI_API_KEY: str = Field("sk-local-placeholder", alias="LLM_OPENAI_API_KEY")
    LLM_MODEL: str = Field("deepseek-8b", alias="LLM_MODEL")

    @model_validator(mode="after")
    def _validate_mongo(self):
        # 如果没开 SSH 隧道，则必须提供 MONGO_URI
        if not self.SSH_TUNNEL and not self.MONGO_URI:
            raise ValueError("MONGO_URI is required when SSH_TUNNEL is disabled")
        # 如果开启 SSH 隧道，允许 MONGO_URI 为空（隧道模块会自己拼接本地端口）
        return self
settings = Settings()