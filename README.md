# FinSight_BackEnd
Intelligent Stock Prediction and Advisory Platform backend

## Base Structure 

```
FinSight_BackEnd
├── app
│   ├── __init__.py
│   ├── adapters
│   │   ├── __init__.py
│   │   ├── db
│   │   │   ├── __init__.py
│   │   │   ├── database_client.py
│   │   │   ├── inmemory.py
│   │   │   ├── mongo_client.py
│   │   │   ├── mongo_repos.py
│   │   │   ├── news_repo.py
│   │   │   ├── pg_profile_repo.py
│   │   │   ├── price_provider_mongo.py
│   │   │   ├── rag_conversation_repo.py
│   │   │   └── user_repo.py
│   │   ├── embeddings
│   │   │   ├── hash_embedder.py
│   │   │   ├── projecting_embedder.py
│   │   │   ├── projectors.py
│   │   │   └── sentence_transformers_embed.py
│   │   ├── fetchers
│   │   │   ├── marketaux_fetcher.py
│   │   │   └── rss_fetcher.py
│   │   ├── llm
│   │   │   └── openai_llm.py
│   │   ├── rag
│   │   │   └── prompt_templates.py
│   │   └── vector
│   │       └── mongo_vector_index.py
│   ├── api
│   │   ├── __init__.py
│   │   └── v1
│   │       ├── __init__.py
│   │       ├── auth_router.py
│   │       ├── debug_router.py
│   │       ├── forecast_router.py
│   │       ├── macro_router.py
│   │       ├── news_router.py
│   │       ├── rag_router.py
│   │       ├── rec_router.py
│   │       ├── stocks_router.py
│   │       └── user_router.py
│   ├── config.py
│   ├── core
│   │   ├── errors.py
│   │   └── middleware.py
│   ├── deps.py
│   ├── domain
│   │   ├── __init__.py
│   │   └── models.py
│   ├── forecasters
│   │   ├── __init__.py
│   │   ├── arima_forecaster.py
│   │   ├── base.py
│   │   ├── dilated_cnn_forecaster.py
│   │   ├── lgbm_forecaster.py
│   │   ├── lstm_forecaster.py
│   │   ├── prophet_forecaster.py
│   │   ├── seq2seq_forecaster.py
│   │   ├── stacked_forecaster.py
│   │   └── transformer_forecaster.py
│   ├── jobs
│   │   └── scheduler.py
│   ├── main.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── enum.py
│   │   ├── exception.py
│   │   ├── models.py
│   │   └── rag_dto.py
│   ├── ports
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   ├── llm.py
│   │   ├── scheduler.py
│   │   ├── storage.py
│   │   └── vector_index.py
│   ├── services
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── forecast_service.py
│   │   ├── ingest_pipeline.py
│   │   ├── macro_service.py
│   │   ├── news_service.py
│   │   ├── rag_service.py
│   │   ├── rec_service.py
│   │   ├── stock_recommender.py
│   │   ├── stock_service.py
│   │   └── user_service.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── healthy.py
│   │   ├── news_seed.py
│   │   ├── security.py
│   │   ├── similarity.py
│   │   └── ticker_mapping.py
│   └── watchlist.json
├── docker-compose.yml
├── environment.yml
├── finsight_keypair.pem
├── pyproject.toml
├── README.md
└── requirements.txt

```

