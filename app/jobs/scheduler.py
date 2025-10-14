# app/jobs/scheduler.py
from __future__ import annotations
from typing import List, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import os

from app.adapters.fetchers.marketaux_fetcher import MarketauxFetcher, MarketauxConfig
from app.utils.ticker_mapping import load_watchlist_simple
from app.services.ingest_pipeline import IngestPipeline
from app.config import settings

# def _split_csv(s: Optional[str]) -> List[str]:
#     if not s:
#         return []
#     return [x.strip() for x in s.split(",") if x.strip()]

# def create_scheduler(app, news_repo, embedder):
#     """
#     创建并返回 APScheduler（不自动启动），提供三个任务：
#     - pull_marketaux_us
#     - pull_marketaux_in
#     - pull_rss_all
#     """
#     scheduler = AsyncIOScheduler(timezone="UTC")

#     # watchlist
#     watch = {}
#     if os.path.exists(getattr(settings, "WATCHLIST_FILE", "")):
#         watch = load_watchlist(settings.WATCHLIST_FILE)

#     pipe = IngestPipeline(news_repo=news_repo, embedder=embedder, watchlist=watch)

#     # Marketaux 设置
#     mcfg = MarketauxConfig(
#         api_key=settings.MARKETAUX_API_KEY,
#         qps=float(getattr(settings, "FETCH_QPS", 0.5)),
#         daily_budget=int(getattr(settings, "DAILY_BUDGET_MARKETAUX", 80)),
#         page_size=20,
#     )
#     mfetch = MarketauxFetcher(mcfg)

#     # RSS 设置
#     rss_us = _split_csv(getattr(settings, "RSS_SOURCES_US", ""))
#     rss_in = _split_csv(getattr(settings, "RSS_SOURCES_IN", ""))
#     rss_all = list(dict.fromkeys(rss_us + rss_in))  # 去重
#     rfetch = RSSFetcher(qps=float(getattr(settings, "RSS_QPS", 1.0)))

#     # 任务实现
#     def _job_marketaux(region: str, symbols: List[str]):
#         raw = mfetch.pull_recent(symbols=symbols, since_hours=6, region=region, max_pages=2)
#         dedup_n, stored = pipe.ingest_dicts(raw)
#         print(f"[JOB] marketaux {region} symbols={len(symbols)} -> ingested {stored}")

#     def _job_rss():
#         raw = rfetch.pull_many(rss_all, limit_per_feed=30)
#         dedup_n, stored = pipe.ingest_dicts(raw)
#         print(f"[JOB] rss all feeds={len(rss_all)} -> ingested {stored}")

#     # 读取股票池按地区划分（可简单用后缀 .NS 归为 IN）
#     us_symbols = [k for k in watch.keys() if not k.endswith(".NS")]
#     in_symbols = [k for k in watch.keys() if k.endswith(".NS")]

#     # 注册任务（cron 可从 .env 配置；默认每小时与每15分钟）
#     cron_marketaux_us = getattr(settings, "CRON_MARKETAUX_US", "0 * * * *")
#     cron_marketaux_in = getattr(settings, "CRON_MARKETAUX_IN", "10 * * * *")
#     cron_rss_all = getattr(settings, "CRON_RSS_ALL", "*/15 * * * *")

#     # APS 的 cron 表达式使用 add_job(..., trigger="cron", minute="*", hour="*")
#     def _apply_cron(job_fn, cron: str, **kwargs):
#         # 仅支持常见五段式：m h dom mon dow
#         m, h, dom, mon, dow = cron.split()
#         scheduler.add_job(job_fn, trigger="cron", minute=m, hour=h, day=dom, month=mon, day_of_week=dow, kwargs=kwargs)

#     _apply_cron(_job_marketaux, cron_marketaux_us, region="us", symbols=us_symbols)
#     _apply_cron(_job_marketaux, cron_marketaux_in, region="in", symbols=in_symbols)
#     _apply_cron(_job_rss, cron_rss_all)

#     return scheduler





def create_scheduler(app, news_repo, embedder):
    """
    创建并返回 APScheduler（不自动启动），提供两个 Marketaux 任务：
    - pull_marketaux_us
    - pull_marketaux_in
    """
    scheduler = AsyncIOScheduler(timezone="UTC")

    watch = []
    if os.path.exists(getattr(settings, "WATCHLIST_FILE", "")):
        watch = load_watchlist_simple(settings.WATCHLIST_FILE)

    # 读取股票池按地区划分（若你仍区分 IN/US 可按后缀 .NS 拆分，否则直接一组）
    us_symbols = [s for s in watch if not s.endswith(".NS")]
    in_symbols = [s for s in watch if s.endswith(".NS")]

    pipe = IngestPipeline(news_repo=news_repo, embedder=embedder, watchlist=watch)

    # Marketaux 设置
    mcfg = MarketauxConfig(
        api_key=settings.MARKETAUX_API_KEY,
        qps=float(getattr(settings, "FETCH_QPS", 0.5)),
        daily_budget=int(getattr(settings, "DAILY_BUDGET_MARKETAUX", 80)),
        page_size=20,
    )
    mfetch = MarketauxFetcher(mcfg)

    # 任务实现
    def _job_marketaux(region: str, symbols: List[str]):
        raw = mfetch.pull_recent(symbols=symbols, since_hours=6, region=region, max_pages=2)
        dedup_n, stored = pipe.ingest_dicts(raw)
        print(f"[JOB] marketaux {region} symbols={len(symbols)} -> ingested {stored}")

    # 读取股票池按地区划分
    us_symbols = [k for k in watch.keys() if not k.endswith(".NS")]
    in_symbols = [k for k in watch.keys() if k.endswith(".NS")]

    # 注册任务（cron 可从 .env 配置；默认每小时）
    cron_marketaux_us = getattr(settings, "CRON_MARKETAUX_US", "0 * * * *")
    cron_marketaux_in = getattr(settings, "CRON_MARKETAUX_IN", "10 * * * *")

    def _apply_cron(job_fn, cron: str, **kwargs):
        # 仅支持常见五段式：m h dom mon dow
        m, h, dom, mon, dow = cron.split()
        scheduler.add_job(job_fn, trigger="cron", minute=m, hour=h, day=dom, month=mon, day_of_week=dow, kwargs=kwargs)

    _apply_cron(_job_marketaux, cron_marketaux_us, region="us", symbols=us_symbols)
    _apply_cron(_job_marketaux, cron_marketaux_in, region="in", symbols=in_symbols)

    return scheduler