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
    
    # —— 新闻抓取主源 & 抓取参数 ——
    MARKETAUX_API_KEY: str = Field("", env="MARKETAUX_API_KEY")  # 可先留空
    WATCHLIST_FILE: str = Field("./watchlist.json", env="WATCHLIST_FILE")
    MARKETAUX_DEFAULT_SYMBOLS: str = Field("", env="MARKETAUX_DEFAULT_SYMBOLS")
    
    FETCH_QPS: float = Field(0.5, env="FETCH_QPS")                    # Marketaux 节流
    DAILY_BUDGET_MARKETAUX: int = Field(80, env="DAILY_BUDGET_MARKETAUX")

    RSS_QPS: float = Field(1.0, env="RSS_QPS")
    RSS_SOURCES_US: str = Field(
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml,https://www.reuters.com/finance/markets/rss",
        env="RSS_SOURCES_US"
    )
    RSS_SOURCES_IN: str = Field(
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms,https://www.livemint.com/rss/markets",
        env="RSS_SOURCES_IN"
    )

    # —— 调度 Cron（分 时 日 月 周）——
    CRON_MARKETAUX_US: str = Field("0 * * * *", env="CRON_MARKETAUX_US")
    CRON_MARKETAUX_IN: str = Field("10 * * * *", env="CRON_MARKETAUX_IN")
    CRON_RSS_ALL: str = Field("*/15 * * * *", env="CRON_RSS_ALL")

    EMBEDDING_PROVIDER: str = Field("openai", env="EMBEDDING_PROVIDER")
    ST_MODEL: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="ST_MODEL")
    DEFAULT_VECTOR_DIM: int = Field(64, env="DEFAULT_VECTOR_DIM")

    PROJECTION_METHOD: str = Field("srp", env="PROJECTION_METHOD")  # srp | none | pca(后续)
    PROJECTION_DIM: int = Field(64, env="PROJECTION_DIM")
    PROJECTION_SEED: int = Field(42, env="PROJECTION_SEED")

    # VECTOR_BACKEND: str = Field("mongo", env="VECTOR_BACKEND")  # mongo | pgvector
    VECTOR_BACKEND: str = Field("pgvector", env="VECTOR_BACKEND")  # mongo | pgvector
    PG_DSN: str = Field("postgresql://richsion@localhost:5432/finsight", env="PG_DSN")
    
    LLM_PROVIDER: str = Field("openai", env="LLM_PROVIDER")
    
    NEWS_FETCH_CRON: str = "0 * * * *"  # 每小时
    # DEFAULT_VECTOR_DIM: int = 32
    
    # 运行环境/调试
    ENV: str = Field("dev", env="ENV")        # dev | prod
    DEBUG: bool = Field(True, env="DEBUG")     # True: 输出更详细错误/开启调试路由

    # v2 风格，代替 class Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
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