from __future__ import annotations
from typing import Dict, List
import json, re, csv, os

def load_watchlist_simple(path: str) -> list[str]:
    """
    读取一个仅含 ticker 列表(JSON 数组)的 watchlist。
    """
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x).strip().upper() for x in data if str(x).strip()]
    # 兼容 { "AAPL": [...], ... } 的旧格式
    if isinstance(data, dict):
        return [k.strip().upper() for k in data.keys()]
    return []

def map_tickers(title: str, text: str, watch: Dict[str, List[str]]) -> List[str]:
    hay = f"{title}\n{text}".lower()
    hits = []
    for tk, aliases in watch.items():
        for a in aliases:
            if not a: 
                continue
            # 简单词边界匹配，避免过度误报
            pat = r"\b" + re.escape(a) + r"\b"
            if re.search(pat, hay):
                hits.append(tk)
                break
    # 去重
    return sorted(list(set(hits)))

# --- 新增/补充: 20维画像映射 ---
# 向量顺序（务必与同事API一致）：
# 0..10: 行业偏好 (11维)
INDUSTRY_INDEX = {
    "utilities": 0, "technology": 1, "consumer_defensive": 2, "healthcare": 3, "basic_materials": 4,
    "real_estate": 5, "energy": 6, "industrials": 7, "consumer_cyclical": 8, "communication_services": 9,
    "financial_services": 10,
}
# 11..19: 投资偏好 (9维) —— 暂时置零或按简易规则打分
INV_COUNT = 9

def _norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")

def topics_to_industry_weights(topics: list[str]) -> list[float]:
    """把新闻 topics/tickers 里的行业词规整成 11维 soft-one-hot。"""
    vec = [0.0]*len(INDUSTRY_INDEX)
    for t in topics or []:
        k = _norm(t)
        # 常见别名归一
        alias = {
            "tech":"technology","it":"technology","software":"technology",
            "semiconductors":"technology",
            "energy":"energy",
            "health_care":"healthcare",
            "consumer_cyclical":"consumer_cyclical",
            "consumer_discretionary":"consumer_cyclical",
            "consumer_defensive":"consumer_defensive",
            "staples":"consumer_defensive",
            "industrials":"industrials",
            "materials":"basic_materials","basic_materials":"basic_materials",
            "real_estate":"real_estate",
            "communication_services":"communication_services",
            "telecom":"communication_services",
            "financials":"financial_services","financial_services":"financial_services",
            "utilities":"utilities",
        }
        k = alias.get(k, k)
        if k in INDUSTRY_INDEX:
            vec[INDUSTRY_INDEX[k]] += 1.0
    # 归一化
    s = sum(vec)
    return [v/s if s>0 else 0.0 for v in vec]

def build_profile20_from_topics_and_signals(topics: list[str], tickers: list[str]) -> list[float]:
    """构造 20d 画像向量：前11维行业，后9维投资偏好(先置0)。"""
    ind11 = topics_to_industry_weights(topics)
    inv9  = [0.0]*INV_COUNT
    return ind11 + inv9