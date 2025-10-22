from __future__ import annotations
from typing import Dict, List, Iterable
import hashlib
import time
import feedparser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

def _safe_pubdate(entry) -> str:
    # 尝试从 published/updated 中解析为 ISO8601（UTC）
    dt = None
    for key in ("published_parsed", "updated_parsed"):
        if getattr(entry, key, None):
            try:
                dt = datetime.fromtimestamp(time.mktime(getattr(entry, key)), tz=timezone.utc)
                break
            except Exception:
                pass
    if dt is None:
        for key in ("published", "updated"):
            val = entry.get(key)
            if val:
                try:
                    dt = parsedate_to_datetime(val).astimezone(timezone.utc)
                    break
                except Exception:
                    pass
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.replace(tzinfo=None).isoformat()

class RSSFetcher:
    """通用 RSS 抓取，输出规范化结构；不含 tickers（需后续映射）"""
    def __init__(self, qps: float = 1.0, timeout: float = 10.0):
        self.qps = qps
        self.timeout = timeout
        self._last_ts = 0.0

    def _throttle(self):
        if self.qps <= 0:
            return
        min_interval = 1.0 / self.qps
        now = time.time()
        sleep = self._last_ts + min_interval - now
        if sleep > 0:
            time.sleep(sleep)
        self._last_ts = time.time()

    def pull_many(self, feeds: Iterable[str], limit_per_feed: int = 30) -> List[Dict]:
        out: List[Dict] = []
        for url in feeds:
            self._throttle()
            parsed = feedparser.parse(url)
            for e in (parsed.entries or [])[:limit_per_feed]:
                link = e.get("link") or ""
                external_id = hashlib.sha1(link.encode("utf-8")).hexdigest()
                title = e.get("title") or ""
                summary = e.get("summary") or e.get("description") or ""
                source = parsed.feed.get("title") or parsed.feed.get("link") or "rss"
                published_at = _safe_pubdate(e)
                out.append({
                    "external_id": external_id,
                    "title": title,
                    "text": summary,
                    "url": link,
                    "source": source,
                    "published_at": published_at,
                    "tickers": [],     # RSS 不带；后续映射
                    "topics": [],
                    "sentiment": None, # 没有就留空
                })
        return out