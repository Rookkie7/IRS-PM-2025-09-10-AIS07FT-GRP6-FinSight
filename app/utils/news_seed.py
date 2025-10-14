from __future__ import annotations
from datetime import datetime, timezone, timedelta

now = datetime.now(timezone.utc)

SEED_NEWS = [
    {
        "news_id": "n1",
        "title": "NVIDIA announces new AI GPU lineup amid strong data center demand",
        "text": "The company highlighted expanding AI partnerships with cloud providers and enterprises.",
        "source": "Reuters",
        "published_at": now - timedelta(hours=2),
        "tickers": ["NVDA"],
        "topics": ["AI", "Semiconductors", "Cloud"],
        "sentiment": 0.6
    },
    {
        "news_id": "n2",
        "title": "Apple reported steady iPhone sales; services segment continues to grow",
        "text": "Analysts expect services revenue to offset hardware cyclicality over the next quarters.",
        "source": "Bloomberg",
        "published_at": now - timedelta(hours=8),
        "tickers": ["AAPL"],
        "topics": ["Consumer Electronics", "Ecosystem"],
        "sentiment": 0.2
    },
    {
        "news_id": "n3",
        "title": "TCS secures multi-year digital transformation deal with European bank",
        "text": "Engagement covers cloud migration, data modernization, and AI-powered analytics.",
        "source": "Economic Times",
        "published_at": now - timedelta(days=1, hours=3),
        "tickers": ["TCS.NS"],
        "topics": ["IT Services", "Cloud", "Banking"],
        "sentiment": 0.4
    }
]
