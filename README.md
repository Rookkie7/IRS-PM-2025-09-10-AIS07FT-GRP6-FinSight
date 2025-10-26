# FinSight: Intelligent Stock Predictionand Advisory Platform

---

## SECTION 1 : PROJECT TITLE
### FinSight: Intelligent Stock Predictionand Advisory Platform

---

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT
At FinSight, we’ve tried to cover all fronts of data science that help a market participant get the big picture at a glance and act fast. We learn each user’s preferences and tailor what we surface: relevant news, an evolving list of interesting stocks, short-horizon predictions on the names that matter, and a specialised LLM that behaves like a personal financial research analyst — reading your docs, explaining its thinking, and showing its work.

All actionable items get value-added annotations: news sentiment tags from a finance-tuned classifier (e.g., FinBERT), a lightweight risk lens on stocks, and a projected stock movement view using deep time-series/recurrent models. For narrative answers, we use RAG so the LLM cites the exact passages it relied on. We emphasise explainability: clear reasoning traces, citations, and intuitive visuals — so a human can audit why a conclusion was reached and what would change it. This helps us in grounding responses in evidence and reducing hallucinations.

On the data side, we stream in news and filings continuously, parse the text, and maintain vector embeddings for users, news items, and tickers alongside the vectors we use for LLM retrieval. We then score relationships with standard similarity measures (cosine/inner-product) so the system can match “you-shaped” interests against fresh content and instruments quickly. This is classic vector search under the hood (FAISS/pgvector), but tuned for markets: fast nearest-neighbour lookups to power news recommendations, stock watchlists, and context retrieval for the LLM.

Operationally, we engineered the stack to be practical: automated ingestion and chunking of messy PDFs/docs, vector search to pull the right context quickly, and a lightweight serving layer that scales from our laptops to GPUs on the cluster. This collaboration grounds the project in real market workflows while giving us hands-on exposure to production-grade financial AI — where reliability, latency, and traceability matter as much as raw accuracy

---

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name  | Student ID (MTech Applicable)  | Work Items (Who Did What) | Email (Optional) |
| :------------ |:---------------:| :-----| :-----|
| **Huo Yiming** | **A0328696J** | xxxxxxxxxx yyyyyyyyyy zzzzzzzzzz|  |
| **Li Jiajun** | **A0326795M** | xxxxxxxxxx yyyyyyyyyy zzzzzzzzzz|  |
| **Samarth Soni** | **A0329960U** | xxxxxxxxxx yyyyyyyyyy zzzzzzzzzz|  |
| **SU Yuxuan** | **A0329926N** | xxxxxxxxxx yyyyyyyyyy zzzzzzzzzz|  |
| **Wang Yixi** | **A0328469M** | xxxxxxxxxx yyyyyyyyyy zzzzzzzzzz| e1547224@u.nus.edu |

---

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

[![Sudoku AI Solver](http://img.youtube.com/vi/-AiYLUjP6o8/0.jpg)](https://youtu.be/-AiYLUjP6o8 "Sudoku AI Solver")

Note: It is not mandatory for every project member to appear in video presentation; Presentation by one project member is acceptable. 
More reference video presentations [here](https://telescopeuser.wordpress.com/2018/03/31/master-of-technology-solution-know-how-video-index-2/ "video presentations")

---

## SECTION 5 : USER GUIDE

`Refer to appendix <Installation & User Guide> in project report at Github Folder: ProjectReport`

### Backend

```shell
git clone https :// github.com/jiajun -lab/ FinSight_BackEnd .git
FinSight_BackEnd
cd FinSight_BackEnd
# Create virtual environment
uv venv
source .venv/bin/ activate

# Install dependencies
uv pip install -e .
# (or pip install -e .)
# Running the Backend
uvicorn app.main:app --reload --port 8000
```

### Frontend

```shell
# 1) Clone this repository
git clone https :// github.com/jiajun -lab/ FinSight_Frontend .git
FinSight_FrontEnd
cd FinSight_FrontEnd

# 2) In this folder :
pnpm i
# 3) set backend endpoint ( optional ; default already localhost ):
echo 'VITE_BACKEND_BASE_URL =http ://127.0.0.1:8000 ' > .env

# 4) run
pnpm dev

# 5) quick restart
rm -rf node_modules pnpm -lock.yaml package -lock.json # optional clean
reboot
pnpm i
pnpm dev
```

### Database

- MongoDB: Stores news and optionally vector embeddings.
- PostgreSQL + pgvector: Stores user profiles, embeddings, and recommendations.

To connect, simply place finsight_keypair.pem in the project root. The program will automatically establish the SSH tunnel and access the databases.

### RAG Component Deployment

1. Deploy Ragflow locally: Follow the official guide at https://ragflow.io/docs/dev/build_docker_image

2. Run local LLM via Ollama:

   ```
   ollama run deepseek -r1:8b
   ```

3. Replace your Ragflow key: Add it to the .env file as:

   ```
   RAGFLOW_API_KEY =<your -key >
   ```

---
## SECTION 6 : PROJECT REPORT / PAPER

`Refer to project report at Github Folder: ProjectReport`



