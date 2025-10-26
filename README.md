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
├── declaration_and_prompts.txt # Our LLM usage disclaimer and prompts history
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

# Git workflow

## Project Setup
```
# Clone the repository
git clone https://github.com/jiajun-lab/FinSight_BackEnd.git

# Navigate into the repo
cd FinSight-QuantLab

# Set up your virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # (on macOS/Linux)
venv\Scripts\activate       # (on Windows)

# Install dependencies
pip install -r requirements.txt
```
## Branching model
| Branch | Contributor | Purpose |
| :------ | :---: | ----: |
| `main`  | PR from develop  | Stable, production-ready branch. Contains reviewed and validated modules.    |
| `develop`   | PR from feature/*  | Integration branch for features currently under development. Regularly merged into `main` once stable.    |
| `feature/your-feature-name` | developers   | Personal working branches for new experiments, models, or scripts. Created from `develop` and merged back via Pull Request.    |

**Example branch names:**
```
feature/distance-approach
feature/pca-enhancement
feature/ml-dbscan
```

## Branching Flow
1.	Pull the latest changes:
```
git fetch --all
git checkout develop
git pull origin develop
# -- OR --
git rebase origin develop
```
2.	Create a new branch:
```
git checkout -b feature/copula-modification
```
3.	Work locally and commit frequently (_see commit rules below_).
4.	Push your branch:
```
git push origin feature/copula-modification
```
5.	Create a Pull Request to develop once done. (_refer to PR checklist below_)

## Commit Guidelines
### Commit Message Convention
**Follow the format:**
```
Module | Component | Short description
```
e.g.
```
Statistical | Distance Approach | Added z-score signal thresholds
Statistical | PCA Approach | Tuned window size and reversion thresholds
ML | DBSCAN | Implemented clustering for mean-reversion candidates
DL | LSTM | First draft implementation
Core Util | Data | Added preprocessing pipeline with forward-fill logic
System | Admin | Reorganised repo structure
```
### Guidelines and Collaboration Notes:
- **Pull or Rebase** before pushing to avoid merge conflicts.
- Keep your feature branches atomic — one branch per experimental approach.
- Document methodology summary and formulae used, parameters, metrics, and results inside each IPYNB
- Commit messages: Use present tense (“Add”, not “Added”); Keep descriptions under ~80 characters.
- **Squash** to group small related edits under one commit


## Pull Request Checklist
Before creating a Pull Request (PR) from your feature/* branch into develop, please go through this checklist.\
This ensures smooth merges, consistent code, and no merge conflicts.\
"No broken commits" -- we want to ensure that every commit that ends up in develop or main is runnable and stable. I.e., the project should *not* be in a half-working state after merging\
With rebase-squash-merge train, we ensure a clean, linear history; merging repeatedly creates unnecessary merge commits that clutter the log
1. Sync with remote to ensure local repo is up to date
```
git fetch --all
# -- OR --
git fetch origin
```
2. Pull or rebase feature branch onto latest `develop`
```
# Switch to develop and update it
git checkout develop
git pull origin develop

# Go back to your feature branch
git checkout feature/your-branch-name

# Rebase your changes on top of latest develop
# Helps to keep commit history linear and avoid messy merges during the PR review
git rebase develop

# If there are conflicts, resolve them, then continue:
git add .
git rebase --continue
```
3. Check commit history & push your updated branch\
_refer to "Squash Commits" section_
4. Verify Code & Tests - check no errors, printed outputs, comments for documentation, no hard coded file paths, helpful commit message, etc.
5. Open PR\
Open the repo on GitHub, create a PR from feature/your-branch-name → develop\
Fill in title, description, and confirmations\
You can self-review changes overview for last cross-check
6. For PR review, another teammate should cross-check the macros and approve via PR comments before merging to develop

### Squash Commits
**NOTE**: Only the _latest / top_ commits on your branch history should be attempted to be squashed.
1.	Identify how many commits you want to squash (e.g., the last 3):
```
git log --oneline

# To find all commits since the last time you pushed to remote
git log origin/feature/your-branch-name..HEAD --oneline

# To find all commits since branching out of develop
git log origin/develop..HEAD --oneline
```
2.	Start an Interactive rebase:
```
git rebase -i HEAD~N # where N is the number of commits to squash

e.g.
git rebase -i HEAD~3
```
3.	In the vim editor that opens, ALWAYS leave the first commit as `pick`, change the rest to `squash` (or `s`)
```
# Enter "insert" mode in vim editor from "normal" mode by clicking "i"
# Escape "insert" mode and back to "normal" by clicking "Esc"
# Save editor and quit by using ":wq" in "normal" mode
# Quit editor without saving by using ":!q"

pick 1234abc Module1 | Component1 | Message 1
s 5678def Module1 | Component1 | Message 2
squash 9abcd12 Module1 | Component1 | Message 3

#Save and close
```
4.	The above will combine the 3 commit hashes into the singular top one you picked. Then, a new editor will open where you can edit the combined commit message from "Module1 | Component1 | Message 1" to "Module1 | Component1 | Message 1+2+3"
5.	Push your changes (force-with-lease to update branch safely):
```
git push origin feature/your-branch-name --force-with-lease

# if history has changed locally you may get an error because some of the squashed commits were already pushed to remote
# --force forgets whatever is on the branch and replaces with the new local history
# --force-with-lease helps to ensure only YOUR remote history is replaced; if someone else committed too, you need to first fetch
git fetch --all

```
### Resolving Conflicts During Rebase -- tutorial
Sometimes, while rebasing your feature branch onto the latest develop, Git will stop and show a message like:\
`CONFLICT (content): Merge conflict in src/statistical/distance.py`\
This just means both branches edited the same lines in the same file. Following is how to deal with this:
```
#List conflicted files
git status
#conflicted files appear as:
#    both modified: src/statistical/distance.py

# Open each conflicted file
# Look for conflict markers:
#    <<<<<<< HEAD
#    # your feature branch version
#    =======
#    # develop branch version
#    >>>>>>> develop
# Manually edit the file to keep or combine the correct lines, and remove the markers.

# Mark conflicts as resolved
git add src/statistical/distance.py

# Continue the rebase.
#This will either take you to the next conflict or print a success message
git rebase --continue

# If you make a mistake and want to restart (This returns you to the branch’s pre-rebase state.):
git rebase --abort

# When rebase finishes successfully, review commit history, squash small commits together into one logical unit, push the updated branch:
git status
git log
git rebase -i HEAD~N
git push origin feature/your-branch-name --force-with-lease

```
Quick tips:
- Resolve conflicts file by file; don’t panic if you see many — just handle them one at a time.
- Use your IDE’s Git conflict tool (VS Code, PyCharm, etc.) — they highlight which side is yours vs theirs.
- After resolving, always run a quick sanity test to ensure the merged file still runs correctly.


## Collaborators:
Jiajun Li\
Yiming Huo\
Wang Yixi\
Su Yuxuan\
Samarth Soni