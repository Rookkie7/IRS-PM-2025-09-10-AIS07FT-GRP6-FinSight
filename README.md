# FinSight_BackEnd
Intelligent Stock Prediction and Advisory Platform backend

## Base Structure 

```
finsight/
  app/
    main.py
    config.py
    deps.py
    api/
      __init__.py
      v1/
        news_router.py
        rec_router.py
        rag_router.py
        forecast_router.py
    domain/
      models.py         # models (News, Stock, EmbeddingVector, Recommendation, ForecastResult）
      enums.py
    services/
      news_service.py
      rec_service.py
      rag_service.py
      forecast_service.py
    ports/              # abstract interface（依赖倒置）
      vector_index.py   # VectorIndexPort
      embedding.py      # EmbeddingProviderPort
      llm.py            # LLMProviderPort
      storage.py        # NewsRepoPort, StockRepoPort
      scheduler.py      # Task scheduling port
    adapters/
      db/
        mongo_client.py
        news_repo_mongo.py
        stock_repo_mongo.py
      vector/
        mongo_vector_index.py   # MongoDB Atlas Vector Search
        pgvector_index.py       # Postgres+pgvector
      embeddings/
        openai_embed.py
        sentence_transformers_embed.py
      llm/
        openai_llm.py
        qwen_llm.py
      tasks/
        celery_app.py
        news_tasks.py
        rec_tasks.py
        forecast_tasks.py
      rag/
        retriever.py
        ranker.py
        prompt_templates.py
    utils/
      log.py
      ids.py
      time.py
      metrics.py
  tests/
    test_news.py
    test_rec.py
  docker/
    docker-compose.yml
  pyproject.toml
  .env.example
  README.md
```

1. **Pyproject.toml:**  项目依赖与构建配置（等同于 requirements.txt + setup），便于 pip install -e . 和统一格式化/静态检查工具。
2. .env: 环境变量
3. Web structure:  FastAPI
4. Deps.py: 依赖注入。把具体实现装进各个Service，便于替换。
5. config.py: 集中管理来自.env的配置
6. api: 接口层
7. domain: 定义业务对象结构与约束
   1. model：数据模型
   2. enums：枚举
8. services: 服务层，只表示业务流程，调用 ports/* 抽象接口，不依赖具体适配器
9. ports：抽象接口，定义需要什么能力，而不是实现，实现在adapter/*
10. Adapters：依赖实现
    1. db：
       - **mongo_client.py**：Motor 客户端与数据库句柄，集中复用连接、配置索引。
       - **news_repo_mongo.py**：NewsRepoPort 的 Mongo 实现（插入、更新嵌入、查询）。
       - **stock_repo_mongo.py**：StockRepoPort 的 Mongo 实现（保存股票基础/向量、检索/过滤）。
    2. vector：
       - **mongo_vector_index.py**：MongoDB Atlas Vector Search 实现 VectorIndexPort
       - **pgvector_index.py**：Postgres+pgvector 的实现（INSERT/UPDATE 向量、<-> 相似度检索）。
    3. embeddings:
       - **openai_embed.py**：调用 OpenAI Embeddings（如 text-embedding-3-small），并做维度对齐到 32（PCA/投影/截断）。
       - **sentence_transformers_embed.py**：本地嵌入（SentenceTransformer），离线可用；同样做 32 维落盘。
    4. Llm:
       - **openai_llm.py**：OpenAI Chat/Completions 封装（温度、系统提示、流式等）。
       - **qwen_llm.py**：Qwen 封装（本地/云端均可），便于切换成本与延迟。
    5. Rag:
       - **retriever.py**：检索器（query→embedding→向量召回→拉原文）。
       - **ranker.py**：重排序（相似度×新鲜度×多样性 MMR；可加新闻可信度/来源权重）。
       - **prompt_templates.py**：统一提示词模板（结构化答案、证据引用、风控提示等）。
    6. Tasks:
       - **celery_app.py**：Celery 初始化（broker/backend、时区、重试策略）。
       - **news_tasks.py**：新闻相关异步任务（定时抓取、批量嵌入、重建索引等）。
       - **rec_tasks.py**：推荐离线任务（用户画像更新、候选池预计算、A/B 实验数据落地）。
       - **forecast_tasks.py**：预测任务（训练/回测/批量生成未来 yhat、缓存刷新）。
11. Utils: 工具



| 模块 (Module)                                | 中文说明                                                     | English Explanation                                          |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Pyproject.toml**                           | 项目依赖与构建配置（等同于 requirements.txt + setup），便于 `pip install -e .`，统一格式化 / 静态检查工具 | Project dependencies and build config (equivalent to `requirements.txt + setup`), enables `pip install -e .`, standardizes formatting / static checks |
| **.env**                                     | 环境变量文件，存储数据库、API Key 等敏感配置                 | Environment variables file, stores DB connections and API keys |
| **Web (FastAPI)**                            | Web 框架层，使用 FastAPI 搭建 RESTful / GraphQL API          | Web framework layer, FastAPI for RESTful / GraphQL APIs      |
| **Deps.py**                                  | 依赖注入，把实现注入 Service，方便替换                       | Dependency injection, binds concrete implementations into Services |
| **config.py**                                | 集中管理来自 `.env` 的配置项                                 | Centralized config management, loads settings from `.env`    |
| **api/**                                     | 接口层，对外提供 HTTP API                                    | API layer, exposes HTTP endpoints                            |
| **domain/**                                  | 业务领域层，定义业务对象结构与约束                           | Domain layer, defines business object structures and constraints |
| ├─ model/                                    | 数据模型定义                                                 | Data model definitions                                       |
| └─ enums/                                    | 枚举类型定义                                                 | Enumeration definitions                                      |
| **services/**                                | 服务层：只表示业务流程，调用 ports/* 抽象接口，不依赖实现    | Service layer: business logic only, calls ports/*, no dependency on implementations |
| **ports/**                                   | 抽象接口，定义需要的能力，具体实现在 adapters/*              | Abstract interfaces defining required capabilities, implemented in adapters/* |
| **adapters/**                                | 依赖实现层，提供 ports 的具体实现                            | Adapter layer, concrete implementations of ports             |
| ├─ db/mongo_client.py                        | Motor 客户端与 DB 句柄，集中复用连接、配置索引               | Motor client and DB handler, reuses connections and configures indexes |
| ├─ db/news_repo_mongo.py                     | `NewsRepoPort` 的 Mongo 实现（插入、更新嵌入、查询）         | Mongo implementation of `NewsRepoPort` (insert, update embeddings, query) |
| ├─ db/stock_repo_mongo.py                    | `StockRepoPort` 的 Mongo 实现（保存股票向量、检索/过滤）     | Mongo implementation of `StockRepoPort` (store stock vectors, search/filter) |
| ├─ vector/mongo_vector_index.py              | MongoDB Atlas Vector Search 实现 `VectorIndexPort`           | MongoDB Atlas Vector Search implementation of `VectorIndexPort` |
| ├─ vector/pgvector_index.py                  | Postgres + pgvector 实现（向量插入/更新，相似度检索）        | Postgres + pgvector implementation (insert/update vectors, similarity search) |
| ├─ embeddings/openai_embed.py                | 调用 OpenAI Embeddings，对齐到 32 维                         | Calls OpenAI Embeddings, aligns to 32 dimensions             |
| ├─ embeddings/sentence_transformers_embed.py | 本地嵌入 (SentenceTransformer)，同样落盘为 32 维             | Local embeddings (SentenceTransformer), reduced to 32 dimensions |
| ├─ llm/openai_llm.py                         | OpenAI Chat/Completions 封装（温度、系统提示、流式输出）     | Wrapper for OpenAI Chat/Completions (temperature, prompts, streaming) |
| ├─ llm/qwen_llm.py                           | Qwen 封装（本地/云端均可），方便切换成本与延迟               | Wrapper for Qwen (local/cloud), trade-off cost and latency   |
| ├─ rag/retriever.py                          | 检索器（query→embedding→召回→拉原文）                        | Retriever (query→embedding→recall→fetch docs)                |
| ├─ rag/ranker.py                             | 重排序（相似度 × 新鲜度 × 多样性 MMR，可加来源权重）         | Re-ranker (similarity × recency × diversity MMR, source weighting) |
| ├─ rag/prompt_templates.py                   | 统一提示词模板（结构化答案、引用、风控提示）                 | Unified prompt templates (structured answers, citations, safety checks) |
| ├─ tasks/celery_app.py                       | Celery 初始化（broker/backend、时区、重试策略）              | Celery initialization (broker/backend, timezone, retries)    |
| ├─ tasks/news_tasks.py                       | 新闻相关异步任务（抓取、嵌入、索引重建）                     | News-related async tasks (crawl, embed, rebuild index)       |
| ├─ tasks/rec_tasks.py                        | 推荐任务（用户画像更新、候选池预计算、A/B 实验）             | Recommendation tasks (profile updates, candidate precomputation, A/B testing) |
| ├─ tasks/forecast_tasks.py                   | 预测任务（训练/回测、生成未来预测、刷新缓存）                | Forecasting tasks (train/backtest, generate forecasts, refresh cache) |
| **utils/**                                   | 工具函数与通用方法                                           | Utility functions and helpers                                |
