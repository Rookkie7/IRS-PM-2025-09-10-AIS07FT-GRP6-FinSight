// import React, { useState } from 'react';
// import { Clock, ExternalLink, Filter, Search } from 'lucide-react';

// interface NewsItem {
//   id: string;
//   title: string;
//   summary: string;
//   source: string;
//   timestamp: string;
//   category: string;
//   sentiment: 'positive' | 'negative' | 'neutral';
//   relevanceScore: number;
// }

// const mockNews: NewsItem[] = [
//   {
//     id: '1',
//     title: 'Fed Signals Potential Rate Cut in Q2 2024',
//     summary: 'Federal Reserve officials hint at monetary policy adjustments amid changing economic conditions...',
//     source: 'Reuters',
//     timestamp: '2 hours ago',
//     category: 'Monetary Policy',
//     sentiment: 'positive',
//     relevanceScore: 0.95
//   },
//   {
//     id: '2',
//     title: 'Tech Stocks Rally on AI Infrastructure Investments',
//     summary: 'Major technology companies announce significant investments in artificial intelligence infrastructure...',
//     source: 'Bloomberg',
//     timestamp: '4 hours ago',
//     category: 'Technology',
//     sentiment: 'positive',
//     relevanceScore: 0.88
//   },
//   {
//     id: '3',
//     title: 'Energy Sector Faces Regulatory Headwinds',
//     summary: 'New environmental regulations could impact traditional energy companies profitability...',
//     source: 'Financial Times',
//     timestamp: '6 hours ago',
//     category: 'Energy',
//     sentiment: 'negative',
//     relevanceScore: 0.82
//   }
// ];

// export const NewsTab: React.FC = () => {
//   const [searchQuery, setSearchQuery] = useState('');
//   const [selectedCategory, setSelectedCategory] = useState('all');

//   const getSentimentColor = (sentiment: string) => {
//     switch (sentiment) {
//       case 'positive': return 'text-green-600 bg-green-50';
//       case 'negative': return 'text-red-600 bg-red-50';
//       default: return 'text-gray-600 bg-gray-50';
//     }
//   };

//   return (
//     <div className="p-6 space-y-6">
//       {/* Header */}
//       <div className="flex items-center justify-between">
//         <h2 className="text-2xl font-bold text-gray-900">Market News</h2>
//         <div className="flex items-center space-x-4">
//           <div className="relative">
//             <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
//             <input
//               type="text"
//               placeholder="Search news..."
//               value={searchQuery}
//               onChange={(e) => setSearchQuery(e.target.value)}
//               className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
//             />
//           </div>
//           <button className="flex items-center space-x-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors">
//             <Filter className="h-4 w-4" />
//             <span>Filter</span>
//           </button>
//         </div>
//       </div>

//       {/* News Feed */}
//       <div className="space-y-4">
//         {mockNews.map((news) => (
//           <div
//             key={news.id}
//             className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow"
//           >
//             <div className="flex items-start justify-between mb-3">
//               <div className="flex items-center space-x-3">
//                 <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor(news.sentiment)}`}>
//                   {news.sentiment.charAt(0).toUpperCase() + news.sentiment.slice(1)}
//                 </span>
//                 <span className="text-sm text-gray-500">{news.category}</span>
//                 <div className="flex items-center text-gray-400">
//                   <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
//                 </div>
//                 <span className="text-sm text-gray-500">Relevance: {Math.round(news.relevanceScore * 100)}%</span>
//               </div>
//               <button className="text-gray-400 hover:text-gray-600">
//                 <ExternalLink className="h-4 w-4" />
//               </button>
//             </div>
            
//             <h3 className="text-lg font-semibold text-gray-900 mb-2 hover:text-blue-600 cursor-pointer">
//               {news.title}
//             </h3>
            
//             <p className="text-gray-600 mb-3 line-clamp-2">
//               {news.summary}
//             </p>
            
//             <div className="flex items-center justify-between text-sm text-gray-500">
//               <div className="flex items-center space-x-4">
//                 <span className="font-medium">{news.source}</span>
//                 <div className="flex items-center space-x-1">
//                   <Clock className="h-3 w-3" />
//                   <span>{news.timestamp}</span>
//                 </div>
//               </div>
//             </div>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// NewsTab.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";

/* ===============================
 * API base / userId helpers
 * =============================== */
function getApiBase(): string {
    const base =
        (window as any).__API_BASE__ ||
        (import.meta as any)?.env?.VITE_API_BASE ||
        "http://127.0.0.1:8000";
    return String(base).replace(/\/+$/, "");
}
// function getUrlParam(name: string): string | null {
//   try {
//     const url = new URL(window.location.href);
//     return url.searchParams.get(name);
//   } catch {
//     return null;
//   }
// }

// function getUserIdFromAuth(): string | null {
//   try {
//     const raw = localStorage.getItem("auth_user");
//     if (!raw) return null;
//     const obj = JSON.parse(raw);
//     return obj?.user_id ? String(obj.user_id) : null;
//   } catch {
//     return null;
//   }
// }

// function resolveUserId(propUserId?: string): string {
//   return (
//     (propUserId && String(propUserId)) ||
//     getUserIdFromAuth() ||
//     getUrlParam("user_id") ||
//     "demo"
//   );
// }


function safeJSON<T = any>(raw: string | null): T | null {
    if (!raw) return null;
    try { return JSON.parse(raw) as T; } catch { return null; }
}

function getUserIdFromAuth(): string | null {
    // 1) localStorage.auth_user => { "user_id": "..." }
    const rawAuth = localStorage.getItem("auth_user");
    const objAuth = safeJSON<any>(rawAuth);
    const idFromAuth =
        (objAuth && typeof objAuth.user_id === "string" && objAuth.user_id.trim()) ?
            objAuth.user_id.trim() : null;
    if (idFromAuth) {
        console.debug("[userId] picked from localStorage.auth_user.user_id ->", idFromAuth);
        return idFromAuth;
    }

    // 2) localStorage.user => { "id": "..." }
    const rawUser = localStorage.getItem("user");
    const objUser = safeJSON<any>(rawUser);
    const idFromUser =
        (objUser && typeof objUser.id === "string" && objUser.id.trim()) ?
            objUser.id.trim() : null;
    if (idFromUser) {
        console.debug("[userId] picked from localStorage.user.id ->", idFromUser);
        return idFromUser;
    }

    // 兜底：没有拿到就返回 null
    return null;
}
function getUrlParam(name: string): string | null {
    try {
        const url = new URL(window.location.href);
        const v = url.searchParams.get(name);
        return v && v.trim() ? v.trim() : null;
    } catch {
        return null;
    }
}
function resolveUserId(propUserId?: string): string {
    const id =
        (propUserId && String(propUserId).trim()) ||
        getUserIdFromAuth() ||
        getUrlParam("user_id") ||
        "demo";

    if (id === "demo") {
        console.warn("[userId] Using fallback 'demo'. Put real id into localStorage.auth_user or localStorage.user");
    } else {
        console.debug("[userId] resolved ->", id);
    }
    return id;
}



/* ===============================
 * Types
 * =============================== */
type NewsItem = {
    news_id: string;
    title: string;
    source?: string;
    published_at?: string;
    tickers?: string[];
    topics?: string[]; // topics[0] 作为 sector
    url?: string;
    score?: number; // 不渲染，仅保留
};
type RecResponse = {
    user_id: string;
    count: number;
    items: NewsItem[];
};

/* ===============================
 * Styles
 * =============================== */
const gridWrap: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 14,
    width: "100%",
    paddingBottom: 64,
};
const card: React.CSSProperties = {
    border: "1px solid #e7e7ef",
    borderRadius: 14,
    padding: 14,
    background: "#fff",
    display: "flex",
    flexDirection: "column",
    minHeight: 200,
    boxShadow: "0 1px 2px rgba(16,24,40,.04)",
};
const title: React.CSSProperties = {
    fontSize: 18,
    fontWeight: 800,
    lineHeight: 1.35,
    color: "#111827",
    marginBottom: 10,
    display: "-webkit-box",
    WebkitLineClamp: 3,
    WebkitBoxOrient: "vertical",
    overflow: "hidden",
    cursor: "pointer",
};
const badgeRow: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 8,
    flexWrap: "wrap",
};
const badge: React.CSSProperties = {
    fontSize: 12,
    borderRadius: 999,
    padding: "4px 10px",
    lineHeight: 1.4,
    border: "1px solid #e5e7eb",
    background: "#fff",
    color: "#111827",
    fontWeight: 700,
};
const metaBottom: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 12,
    flexWrap: "wrap",
    color: "#6b7280",
    fontSize: 12,
    marginTop: "auto", // 贴底
};
const actions: React.CSSProperties = {
    display: "flex",
    justifyContent: "flex-end",
    marginTop: 12,
    gap: 14,
};
const iconBtn: React.CSSProperties = {
    fontSize: 18,
    cursor: "pointer",
    userSelect: "none",
};
const footerBar: React.CSSProperties = {
    position: "sticky",
    bottom: 0,
    left: 0,
    right: 0,
    background: "#fff",
    borderTop: "1px solid #e5e7eb",
    padding: "10px 12px",
    display: "grid",
    gridTemplateColumns: "1fr auto 1fr",
    alignItems: "center",
    zIndex: 5,
};
const leftZone: React.CSSProperties = { justifySelf: "start" };
const midZone: React.CSSProperties = {
    justifySelf: "center",
    display: "flex",
    alignItems: "center",
    gap: 8,
};
const rightZone: React.CSSProperties = { justifySelf: "end" };
const btn: React.CSSProperties = {
    padding: "8px 12px",
    borderRadius: 8,
    border: "1px solid #ddd",
    background: "#fafafa",
    cursor: "pointer",
};
const pageInput: React.CSSProperties = {
    width: 56,
    padding: "6px 8px",
    borderRadius: 8,
    border: "1px solid #ddd",
    outline: "none",
    textAlign: "center",
};

/* ===============================
 * Helpers
 * =============================== */
function prettySource(raw?: string): string {
    if (!raw) return "Unknown";
    try {
        const url = raw.startsWith("http") ? raw : `https://${raw}`;
        const host = new URL(url).hostname.toLowerCase();
        const map: Record<string, string> = {
            "finance.yahoo.com": "Yahoo Finance",
            "investors.com": "Investor's Business Daily",
            "seekingalpha.com": "Seeking Alpha",
            "zerohedge.com": "ZeroHedge",
            "abcnews.go.com": "ABC News",
            "thestockmarketwatch.com": "StockMarketWatch",
            "businessinsider.com": "Business Insider",
            "barrons.com": "Barron's",
            "bloomberg.com": "Bloomberg",
            "reuters.com": "Reuters",
            "rttnews.com": "RTTNews",
        };
        if (map[host]) return map[host];
        const m = host.match(/([a-z0-9-]+)\.(?:co\.)?(com|org|net|news|finance|ai|io)$/i);
        if (m?.[1]) {
            const brand = m[1].replace(/-/g, " ");
            return brand.charAt(0).toUpperCase() + brand.slice(1);
        }
        return host;
    } catch {
        return raw;
    }
}
const SECTOR_COLORS: Record<string, string> = {
    "Communication Services": "#06b6d4",
    "Consumer Cyclical": "#f97316",
    "Consumer Defensive": "#0ea5e9",
    Energy: "#ef4444",
    "Financial Services": "#22c55e",
    Healthcare: "#a855f7",
    Industrials: "#64748b",
    Materials: "#84cc16",
    "Real Estate": "#d946ef",
    Technology: "#2563eb",
    Utilities: "#14b8a6",
};
function sectorColor(name?: string): string {
    if (!name) return "#111827";
    return SECTOR_COLORS[name] || "#111827";
}
function uniqueByNewsId(list: NewsItem[]): NewsItem[] {
    const seen = new Set<string>();
    const out: NewsItem[] = [];
    for (const it of list) {
        const id = it?.news_id;
        if (!id || seen.has(id)) continue;
        seen.add(id);
        out.push(it);
    }
    return out;
}
function shuffleInPlace<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

/* ===============================
 * 简单缓存：sessionStorage
 * =============================== */
const CACHE_KEY = (uid: string) => `news_pages_v1:${uid}`;
type PagesCache = {
    pages: NewsItem[][];
    pageIndex: number;
    seenIds: string[];
    ts: number; // 存储时间戳，便于设定过期策略
};
function saveCache(uid: string, pages: NewsItem[][], pageIndex: number, seenIds: Set<string>) {
    try {
        const payload: PagesCache = {
            pages,
            pageIndex,
            seenIds: Array.from(seenIds),
            ts: Date.now(),
        };
        sessionStorage.setItem(CACHE_KEY(uid), JSON.stringify(payload));
    } catch {}
}
function loadCache(uid: string): PagesCache | null {
    try {
        const raw = sessionStorage.getItem(CACHE_KEY(uid));
        if (!raw) return null;
        const obj = JSON.parse(raw) as PagesCache;
        // 可选：30 分钟过期
        const THIRTY_MIN = 30 * 60 * 1000;
        if (Date.now() - (obj.ts || 0) > THIRTY_MIN) return null;
        if (!Array.isArray(obj.pages) || obj.pages.length === 0) return null;
        return obj;
    } catch {
        return null;
    }
}

/* ===============================
 * Component
 * =============================== */
export default function NewsTab({ userId: propUserId }: { userId?: string }) {
    const API = getApiBase();
    const userId = resolveUserId(propUserId);

    const [pages, setPages] = useState<NewsItem[][]>([]);
    const [pageIndex, setPageIndex] = useState<number>(0);
    const items = useMemo(() => (pages[pageIndex] || []).slice(0, 9), [pages, pageIndex]); // 严格 9

    const [seenIds, setSeenIds] = useState<Set<string>>(new Set());
    const [liked, setLiked] = useState<Set<string>>(new Set());
    const [bookmarked, setBookmarked] = useState<Set<string>>(new Set());
    const [opLock, setOpLock] = useState<Record<string, boolean>>({});
    const [jump, setJump] = useState<string>("");

    const prefetchingRef = useRef(false); // 防止重复预取
    const mountedRef = useRef(false);

    /* ---------- 首屏：优先从缓存复原；否则从 DB 拿 9 条 ---------- */
    useEffect(() => {
        const cached = loadCache(userId);
        if (cached) {
            console.debug("[cache] restored:", {
                pages: cached.pages.length,
                pageIndex: cached.pageIndex,
                seenIds: cached.seenIds.length,
            });
            setPages(cached.pages);
            setPageIndex(Math.min(cached.pageIndex, cached.pages.length - 1));
            setSeenIds(new Set(cached.seenIds));
            // 复原后立刻预取下一页（无缝体验）
            setTimeout(() => prefetchNext(), 0);
        } else {
            // 无缓存：走数据库（refresh=0），随机 9 条
            void fetchPageFromDB({ initialRandom: true }).then(() => {
                // 首屏准备好后立即开始预取下一页
                prefetchNext();
            });
        }
        mountedRef.current = true;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [userId, API]);

    /* ---------- 每次 pages/pageIndex/seenIds 变动就刷新缓存 ---------- */
    useEffect(() => {
        if (!mountedRef.current) return;
        saveCache(userId, pages, pageIndex, seenIds);
    }, [userId, pages, pageIndex, seenIds]);

    /* ---------- HTTP ---------- */
    async function jsonGet<T>(url: string): Promise<T> {
        console.debug("[GET]", url);
        const r = await fetch(url);
        if (!r.ok) throw new Error(`GET ${url} -> ${r.status}`);
        const ct = r.headers.get("content-type") || "";
        if (!ct.includes("application/json")) {
            const txt = await r.text();
            throw new Error(`Non-JSON response (content-type: ${ct}).\nPreview: ${txt.slice(0, 160)}`);
        }
        return (await r.json()) as T;
    }
    async function jsonPost<T = any>(url: string, body: any): Promise<T> {
        const r = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            keepalive: true,
        });
        const data = (await r.json().catch(() => ({}))) as any;
        if (!r.ok) throw new Error(`POST ${url} -> ${r.status}: ${JSON.stringify(data)}`);
        return data as T;
    }

    /* ---------- 只从 DB 取候选：refresh=0 ---------- */
    async function fetchPageFromDB(params: { initialRandom?: boolean } = {}) {
        const { initialRandom } = params;
        const LIMIT = 9;
        const exHours = 720;

        // 为保证“每次都有 9 条”，取一个较大的 limit 然后本地筛
        const url = `${API}/rec/user/news?user_id=${encodeURIComponent(
            userId
        )}&limit=49&refresh=0&exclude_hours=${exHours}`;
        console.debug("[rec-db] fetch:", url);

        const data = await jsonGet<RecResponse>(url);
        let pool = uniqueByNewsId(
            (data.items || []).filter((it) => it && it.news_id && !seenIds.has(it.news_id))
        );

        if (initialRandom) shuffleInPlace(pool);

        const pageList = pool.slice(0, LIMIT); // 严格 9

        const newPages = pages.slice(0, pageIndex + 1);
        newPages.push(pageList);
        setPages(newPages);
        setPageIndex(newPages.length - 1);

        const newSeen = new Set(seenIds);
        pageList.forEach((it) => newSeen.add(it.news_id));
        setSeenIds(newSeen);

        console.debug(`[rec-db] page ready -> show=${pageList.length}, total pages=${newPages.length}`);
    }

    /* ---------- 后台预取下一页（无缝 Next） ---------- */
    async function prefetchNext() {
        if (prefetchingRef.current) return;
        prefetchingRef.current = true;
        try {
            // 如果“下一页”已经存在，不需要预取
            if (pages[pageIndex + 1]?.length) return;

            // 以“当前已 seen 的集合”为基准做一次“静默 fetch”
            const LIMIT = 9;
            const exHours = 720;
            const seenSnap = new Set(seenIds);

            const url = `${API}/rec/user/news?user_id=${encodeURIComponent(
                userId
            )}&limit=49&refresh=0&exclude_hours=${exHours}`;
            console.debug("[rec-db] prefetch:", url);

            const data = await jsonGet<RecResponse>(url);
            let pool = uniqueByNewsId(
                (data.items || []).filter((it) => it && it.news_id && !seenSnap.has(it.news_id))
            );
            const pageList = pool.slice(0, LIMIT);

            if (pageList.length > 0) {
                setPages((old) => {
                    const cp = old.slice(0);
                    // 只有“当前页是最后一页”时，才把预取结果接在后面
                    const isLast = (cp.length - 1) === pageIndex;
                    if (isLast) cp.push(pageList);
                    return cp;
                });
                setSeenIds((old) => {
                    const n = new Set(old);
                    pageList.forEach((it) => n.add(it.news_id));
                    return n;
                });
                console.debug(`[rec-db] prefetched next page: ${pageList.length}`);
            }
        } catch (e) {
            console.warn("[rec-db] prefetch failed:", e);
        } finally {
            prefetchingRef.current = false;
        }
    }

    /* ---------- 翻页 & 跳页 ---------- */
    function handlePrev() {
        if (pageIndex <= 0) return;
        setPageIndex((i) => i - 1);
        console.debug("[pager] prev ->", pageIndex - 1);
    }
    function handleNext() {
        // 有缓存的下一页 -> 直接切
        if (pages[pageIndex + 1]?.length) {
            setPageIndex((i) => i + 1);
            console.debug("[pager] next -> cached page", pageIndex + 1);
            // 切过去后继续后台预取下一页
            setTimeout(() => prefetchNext(), 0);
            return;
        }
        // 没有缓存 -> 立即触发一次 fetch，并马上预取下一页
        void fetchPageFromDB().then(() => {
            setTimeout(() => prefetchNext(), 0);
        });
    }
    function handleJump() {
        const n = Math.max(1, Math.floor(Number(jump)));
        if (!Number.isFinite(n)) return;
        const target = n - 1;
        if (target < 0 || target >= pages.length) {
            alert(`Page ${n} not available. You currently have ${pages.length} page(s).`);
            return;
        }
        setPageIndex(target);
        // 跳转后同样预取下一页
        setTimeout(() => prefetchNext(), 0);
        console.debug("[pager] jump ->", target);
    }

    /* ---------- 事件：点击 / like / bookmark（维持原有快响应） ---------- */
    function handleOpen(item: NewsItem) {
        // 立即打开
        if (item.url) window.open(item.url, "_blank", "noopener,noreferrer");

        // 非阻塞上报
        const url = `${API}/users/event/click`;
        const payload = {
            user_id: userId,
            news_id: item.news_id,
            dwell_ms: 0,
            liked: false,
            bookmarked: false,
        };
        try {
            const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
            const ok = (navigator as any)?.sendBeacon?.(url, blob);
            if (ok) {
                console.debug("[click] sendBeacon OK", payload);
                return;
            }
        } catch {}
        fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            keepalive: true,
        })
            .then((r) => r.json().catch(() => ({})))
            .then((d) => console.debug("[click] fetch OK", d))
            .catch((e) => console.warn("[click] fetch failed", e));
    }

    function toggleAction(type: "like" | "bookmark", item: NewsItem) {
        const key = `${type}:${item.news_id}`;
        if (opLock[key]) return; // 防抖
        setOpLock((s) => ({ ...s, [key]: true }));

        const setState = type === "like" ? setLiked : setBookmarked;
        const getState = type === "like" ? liked : bookmarked;

        // 乐观更新
        const on = getState.has(item.news_id);
        const op: "add" | "remove" = on ? "remove" : "add";
        const backup = new Set(getState);
        const next = new Set(getState);
        if (on) next.delete(item.news_id);
        else next.add(item.news_id);
        setState(next);

        const url = type === "like" ? `${API}/users/event/like` : `${API}/users/event/bookmark`;
        const payload = { user_id: userId, news_id: item.news_id, op };
        console.debug("[EVENT]", type, "payload =", payload);

        jsonPost(url, payload)
            .then((res) => console.debug(`[${type}] ok:`, res))
            .catch((e) => {
                console.error(`[${type}] ${op} failed`, e);
                setState(backup); // 回滚
            })
            .finally(() => {
                setOpLock((s) => {
                    const n = { ...s };
                    delete n[key];
                    return n;
                });
            });
    }

    /* ---------- 卡片 ---------- */
    function renderCard(it: NewsItem) {
        const likedOn = liked.has(it.news_id);
        const bkOn = bookmarked.has(it.news_id);
        const sector = (it.topics || [])[0] || "";
        const col = sectorColor(sector);

        const tickersColored = (it.tickers || [])
            .slice(0, 3)
            .map((t, i) => (
                <span key={`${it.news_id}:tk:${i}`} style={{ color: col, fontWeight: 800, marginLeft: 6 }}>
          [{(t || "").toUpperCase()}]
        </span>
            ));

        return (
            <div key={it.news_id} style={card}>
                {/* 标题（可点） + 彩色 tickers */}
                <div style={title} title={it.title} onClick={() => handleOpen(it)}>
                    {it.title}
                    {tickersColored}
                </div>

                {/* sector 徽标（字体与边框同色） */}
                <div style={badgeRow}>
                    {sector && (
                        <span style={{ ...badge, color: col, borderColor: col }}>{sector}</span>
                    )}
                </div>

                {/* 底部 meta：Source + 发布时间（贴底） */}
                <div style={metaBottom}>
                    <span>Source: {prettySource(it.source)}</span>
                    <span>·</span>
                    <span>{it.published_at || ""}</span>
                </div>

                {/* 操作按钮：♥ / ☆ */}
                <div style={actions}>
          <span
              style={{
                  ...iconBtn,
                  color: likedOn ? "#f5b301" : "#9ca3af",
                  transform: "translateY(-1px)",
              }}
              title={likedOn ? "Unlike" : "Like"}
              onClick={() => toggleAction("like", it)}
          >
            {likedOn ? "♥" : "♡"}
          </span>
                    <span
                        style={{ ...iconBtn, color: bkOn ? "#f5b301" : "#9ca3af" }}
                        title={bkOn ? "Remove bookmark" : "Bookmark"}
                        onClick={() => toggleAction("bookmark", it)}
                    >
            {bkOn ? "★" : "☆"}
          </span>
                </div>
            </div>
        );
    }

    /* ---------- 渲染 ---------- */
    return (
        <div style={{ padding: 12 }}>
            <div style={gridWrap}>
                {items.slice(0, 9).map((it) => renderCard(it))}
                {items.length === 0 && (
                    <div style={{ gridColumn: "1 / -1", textAlign: "center", color: "#888" }}>
                        No items.
                    </div>
                )}
            </div>

            <div style={footerBar}>
                <div style={leftZone}>
                    <button style={btn} disabled={pageIndex <= 0} onClick={handlePrev}>
                        ← Previous
                    </button>
                </div>

                <div style={midZone}>
          <span style={{ color: "#6b7280", fontSize: 12 }}>
            Page {pageIndex + 1} / {Math.max(1, pages.length || 1)}
          </span>
                    <input
                        type="number"
                        min={1}
                        value={jump}
                        onChange={(e) => setJump(e.target.value)}
                        placeholder="Go to…"
                        style={pageInput}
                    />
                    <button style={btn} onClick={handleJump}>
                        Go
                    </button>
                </div>

                <div style={rightZone}>
                    <button style={btn} onClick={handleNext}>
                        Next →
                    </button>
                </div>
            </div>
        </div>
    );
}

export { NewsTab };


