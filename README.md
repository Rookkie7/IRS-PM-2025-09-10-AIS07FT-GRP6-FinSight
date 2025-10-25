# FinSight Backend

FinSight is an AI-powered financial analytics backend that provides news aggregation, vector-based retrieval, stock forecasting, recommendation, and RAG research assistance — all served through a modular FastAPI architecture.

This repository hosts the backend micro-services, which include:
- **News Fetching & Aggregation**
- **Stock Trend Forecasting (ARIMA / Prophet / LSTM / LightGBM / Transformer / Stacked)**
- **RAG-based Financial Research Analyst**
- **Content-based & Vector-based Stock Recommendation**
- **User Profiles, Watchlists, and Authentication**
- **MongoDB + Postgres (pgvector) Support**

## **Project Structure**

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
 All dependencies are now fully managed by **`pyproject.toml`**.

 ---

## **1. Prerequisites**

| Tool | Requirement |
|---------|------------|
| Python | `3.10+` |
| Docker | for Knowledge Base of Rag which need the  `MongoDB + Postgres` |
| uv / pip | for dependency installation |
| Ragflow | for Rag deployment|
| Ollama| for LLM Provider|

1. Install `uv` (recommended, fastest):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Access the database deployed on EC2 server
This server deployed our MongoDB and PostgresSQL
- `MongoDB` (news storage, vector store optional)
- `PostgreSQL + pgvector` (user profiles, embeddings, recommendations)

just put `finsight_keypair.pem` under the root directory and start the program will automatically connected to our server and access the data.

---

### optional for RAG
Because of our server has limited memory which is not sufficient for deployed on that server, we deployed this in local computer, if you want to check the rag performance, just follow the rest steps. this is only for the RAG part which will not block other functions.
1. Locally deploy `Ragflow` please follow this guide: `https://ragflow.io/docs/dev/build_docker_image` or simply email me 😊(e1538626@u.nus.edu)

2. locally run deepseek-r1:8b in ollama
```
1. Download Ollama: Visit the Ollama website(ollama.com) and download the installation package based on your operating system (Windows, macOS, or Linux).
2. Check installation: `ollama --version`
3. Run local LLM: `ollama run deepseek-r1:8b`
```

3. replace your Ragflow-key in .env file `RAGFLOW_API_KEY=`


## **2. Installation**

```bash
git clone <your-repo-url> FinSight_BackEnd
cd FinSight_BackEnd

# Create virtual environment
uv venv
source .venv/bin/activate

# Install backend dependencies
uv pip install -e .
# (or pip install -e .)
```


> ⚠️ Note: `prophet` will compile CmdStan at first run (can take several minutes).  

---

## ▶️ **3. Start the Backend**

**Note: change your `.env` file according to your settings.**

```bash
uvicorn app.main:app --reload --port 8000
```

API Docs available at:

```
http://localhost:8000/docs
```
The backend project will run on `http://localhost:8000` and the frontend will get all data from this cite.


---

