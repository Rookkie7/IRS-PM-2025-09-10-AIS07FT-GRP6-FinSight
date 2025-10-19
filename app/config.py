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
    LOCAL_BIND_HOST: str | None = None

    # Mongo
    REMOTE_MONGO_HOST: str = "127.0.0.1"
    REMOTE_MONGO_PORT: int = 27017
    LOCAL_MONGO_HOST: str = "127.0.0.1"
    LOCAL_MONGO_PORT: int = 0  # 0=随机

    MONGO_URI: str | None = Field(default=None, alias="MONGO_URI")
    MONGO_DB: str = "finsight"

    # Postgre
    REMOTE_PG_HOST: str = "127.0.0.1"
    REMOTE_PG_PORT: int = 27017
    LOCAL_PG_HOST: str = "127.0.0.1"
    LOCAL_PG_PORT: int = 0  # 0=随机
    POSTGRES_URI: str = Field(..., env="POSTGRES_URI")
    POSTGRES_USER: str = Field(..., env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(..., env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field(..., env="POSTGRES_DB")

    EMBEDDING_PROVIDER: str = Field("openai", env="EMBEDDING_PROVIDER")
    VECTOR_BACKEND: str = Field("pgvector", env="VECTOR_BACKEND")  # mongo | pgvector
    NEWS_FETCH_CRON: str = "0 * * * *"  # 每小时
    DEFAULT_VECTOR_DIM: int = 32

    # LLM Setting
    LLM_PROVIDER: str = Field("deepseek_openai", alias="LLM_PROVIDER")
    LLM_OPENAI_BASE: str = Field("http://127.0.0.1:8000/v1", alias="LLM_OPENAI_BASE")
    LLM_OPENAI_API_KEY: str = Field("sk-local-placeholder", alias="LLM_OPENAI_API_KEY")
    LLM_MODEL: str = Field("deepseek-8b", alias="LLM_MODEL")

    # Rag flow
    RAGFLOW_BASE_URL: str = Field(..., env="RAGFLOW_BASE_URL")  # RAGFlow 的 HTTP 服务地址
    RAGFLOW_API_KEY: str = Field(..., env="RAGFLOW_API_KEY")
    RAGFLOW_MODEL: str = Field(..., env="RAGFLOW_MODEL")
    RAGFLOW_APP_ID: str = ""  # RAGFlow 应用/流水线ID（如需要）
    RAGFLOW_TIMEOUT: int = 60  # 超时时间(s)
    RAG_MAX_HISTORY_TURNS: int = 8  # 多轮记忆条数上限（每次取尾部N轮）
    RAGFLOW_DATASET_IDS: str = Field("", env="RAGFLOW_DATASET_IDS")  # 逗号分隔
    RAGFLOW_CHAT_NAME_PREFIX: str = Field("chat", env="RAGFLOW_CHAT_NAME_PREFIX")

    @model_validator(mode="after")
    def _validate_mongo(self):
        # 如果没开 SSH 隧道，则必须提供 MONGO_URI
        if not self.SSH_TUNNEL and not self.MONGO_URI:
            raise ValueError("MONGO_URI is required when SSH_TUNNEL is disabled")
        # 如果开启 SSH 隧道，允许 MONGO_URI 为空（隧道模块会自己拼接本地端口）
        return self
settings = Settings()