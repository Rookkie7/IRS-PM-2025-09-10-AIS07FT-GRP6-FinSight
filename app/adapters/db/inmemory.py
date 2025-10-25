# app/repositories/inmemory.py
from __future__ import annotations
from typing import Dict, List, Iterable
from app.domain.models import NewsItem, UserProfile, BehaviorEvent

class InMemoryNewsRepo:
    def __init__(self):
        self._store: Dict[str, NewsItem] = {}
    def clear(self):
        self._store.clear()
    def upsert_many(self, items: Iterable[NewsItem]):
        for it in items:
            self._store[it.news_id] = it
    def all(self) -> List[NewsItem]:
        return list(self._store.values())
    def get(self, news_id: str) -> NewsItem | None:
        return self._store.get(news_id)
    def clear(self):
        self._store.clear()
    def ping_detail(self) -> tuple[bool, str | None]:
        return True, None
    def ping(self) -> bool:
        return True
    def latest(self, limit: int = 200) -> List[NewsItem]:
        arr = list(self._store.values())
        arr.sort(key=lambda x: x.published_at, reverse=True)
        return arr[:int(limit)]
    
class InMemoryProfileRepo:
    def __init__(self, dim: int = 32):              # ← 新增
        self._store: Dict[str, UserProfile] = {}
        self._dim = dim                              # ← 新增

    def ping_detail(self) -> tuple[bool, str | None]:
        return True, None
    def ping(self) -> bool:
        return True
    
    def get_or_create(self, user_id: str) -> UserProfile:
        if user_id not in self._store:
            # 用当前嵌入器维度初始化零向量
            self._store[user_id] = UserProfile(user_id=user_id, vector=[0.0]*self._dim)
        return self._store[user_id]

    def save(self, prof: UserProfile):
        self._store[prof.user_id] = prof

class InMemoryEventRepo:
    def __init__(self):
        self._events: List[BehaviorEvent] = []
    def add(self, ev: BehaviorEvent):
        self._events.append(ev)
    def all(self) -> List[BehaviorEvent]:
        return list(self._events)
    def ping_detail(self) -> tuple[bool, str | None]:
        return True, None
    def ping(self) -> bool:
        return True