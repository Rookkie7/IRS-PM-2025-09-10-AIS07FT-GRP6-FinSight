import React, { useEffect, useMemo, useState } from "react";
import { TrendingUp, TrendingDown, Calendar, Target, AlertCircle, BarChart, RefreshCcw, Loader2 } from "lucide-react";
import { fetchForecastBatch } from "../../api/forecast";
// 仅保留类型引用；不再使用 toPredictionData
import type { PredictionData } from "../../utils/forecastMapper";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea
} from "recharts";
import { useAuth } from "../Auth/AuthContext";

const DEFAULT_SYMBOLS = ["AAPL", "TSLA", "NVDA"];
const companyNames = ["Apple Inc.", "Tesla Inc.", "NVIDIA Corp.", "Microsoft Corp."];

// 最近7天收盘价
const API = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function fetchForecastDiagnostics(symbol: string, method: string) {
  const qs = new URLSearchParams({
    method,
    horizons: "1,2,3,4,5,6,7",
    window: "90",
  }).toString();
  const res = await fetch(`${API}/forecast/diagnostics/${encodeURIComponent(symbol)}?${qs}`);
  if (!res.ok) throw new Error("Failed to load diagnostics");
  return res.json(); // { accuracy, risk_level, best_timeframe, data_points, ...}
}

async function fetchLast7Prices(symbol: string) {
  const res = await fetch(`${API}/forecast/prices7?ticker=${encodeURIComponent(symbol)}`);
  if (!res.ok) throw new Error("Failed to load last 7 prices");
  return res.json(); // { prices: Array<{ date: string; close: number }> }
}
// ===== Page Cache Utils =====
type Cached<T> = { at: number; data: T };
const TTL_MS = 5 * 60 * 1000; // 5分钟

function isFresh(at: number, ttl = TTL_MS) {
  return Date.now() - at < ttl;
}

// sessionStorage helpers
function ssRead<T = any>(key: string, ttl = TTL_MS): T | null {
  try {
    const raw = sessionStorage.getItem(key);
    if (!raw) return null;
    const obj = JSON.parse(raw) as Cached<T>;
    if (!obj || typeof obj.at !== "number") return null;
    return isFresh(obj.at, ttl) ? obj.data : null;
  } catch {
    return null;
  }
}
function ssWrite<T = any>(key: string, data: T) {
  try {
    sessionStorage.setItem(key, JSON.stringify({ at: Date.now(), data } satisfies Cached<T>));
  } catch {}
}

// 请求级缓存（内存）
const reqCache = new Map<string, Cached<any>>();
async function getJSONCached(url: string, ttl = TTL_MS) {
  const hit = reqCache.get(url);
  if (hit && isFresh(hit.at, ttl)) return hit.data;
  const res = await fetch(url, { credentials: "include" });
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  const data = await res.json();
  reqCache.set(url, { at: Date.now(), data });
  return data;
}

// 页面状态持久化（选中的 symbol/method/limit）
const UI_STATE_KEY = "predict-ui-state-v1";
type UIState = { symbol: string; method: string; limit: number };

type BatchKey = string;
const CACHE_TTL_MS = 5 * 60 * 1000;
const batchCache = new Map<BatchKey, { at: number; data: PredictionData[] }>();
const inFlight = new Map<BatchKey, Promise<PredictionData[]>>();
const PERSIST_KEY = "predict-batch-cache-v1";
(function hydrateCacheFromSession() {
  try {
    const raw = sessionStorage.getItem(PERSIST_KEY);
    if (!raw) return;
    const obj = JSON.parse(raw) as Record<string, { at: number; data: PredictionData[] }>;
    Object.entries(obj).forEach(([k, v]) => batchCache.set(k, v));
  } catch {}
})();
function persistCacheToSession() {
  try {
    const obj: Record<string, { at: number; data: PredictionData[] }> = {};
    batchCache.forEach((v, k) => (obj[k] = v));
    sessionStorage.setItem(PERSIST_KEY, JSON.stringify(obj));
  } catch {}
}
// 推荐股票（使用页面登录的 userId）
export async function fetchRecommendedSymbols(userId = "demo", topK = 10) {
  const url = `${API}/stocks/recommend?user_id=${encodeURIComponent(userId)}&top_k=${topK}`;
  const res = await fetch(url, { credentials: "include" });
  if (!res.ok) throw new Error("Failed to load recommended stocks");
  const data = await res.json();
  const recs = Array.isArray(data?.recommendations) ? data.recommendations : [];
  // 后端可能是 {ticker} 或 {symbol}
  return recs.map((r: any) => r.ticker ?? r.symbol).filter(Boolean);
}


// ====== 小工具 ======
function relativeFromNow(iso: string) {
  const sec = Math.max(1, Math.floor((Date.now() - new Date(iso).getTime()) / 1000));
  if (sec < 60) return `${sec} seconds ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} minutes ago`;
  const hr = Math.floor(min / 60);
  return `${hr} hours ago`;
}



function normalizeBatchPayload(payload: any): any[] {
  if (!payload) return [];
  if (Array.isArray(payload)) return payload;

  const candidates = [payload.items, payload.data, payload.results, payload.forecasts, payload.list, payload.records, payload.batch];
  for (const c of candidates) if (Array.isArray(c)) return c;

  // 以 symbol 为 key 的对象
  if (payload && typeof payload === "object") {
    const obj =
      (payload.data && !Array.isArray(payload.data) && payload.data) ||
      (payload.batch && !Array.isArray(payload.batch) && payload.batch) ||
      payload;
    if (obj && typeof obj === "object" && !Array.isArray(obj)) {
      return Object.values(obj).flat();
    }
  }
  return [];
}

// === normalizeForecast：统一后端各种结构到 [{date,value}] ===
function normalizeForecast(raw: any): Array<{ date: string; value: number }> {
  if (!raw) return [];
  const take = (arr: any[]) =>
    arr
      .map((p: any) => ({
        date: p.date ?? p.ts ?? p.t ?? p.ds ?? p[0],
        value: p.value ?? p.price ?? p.v ?? p.yhat ?? p.y_pred ?? p[1],
        type: (p.type ?? p.kind ?? p.flag ?? "").toString().toLowerCase(),
      }))
      .filter((p) => p.date && Number.isFinite(p.value));

  if (Array.isArray(raw.points)) {
    const arr = take(raw.points);
    const future = arr.filter((p) => ["pred", "forecast", "future"].includes(p.type));
    return (future.length ? future : arr).map(({ date, value }) => ({ date, value }));
  }
  if (Array.isArray(raw.forecast)) return take(raw.forecast).map(({ date, value }) => ({ date, value }));
  if (Array.isArray(raw.forecast_points)) return take(raw.forecast_points).map(({ date, value }) => ({ date, value }));
  if (Array.isArray(raw.result)) return take(raw.result).map(({ date, value }) => ({ date, value }));
  if (Array.isArray(raw.preds)) return take(raw.preds).map(({ date, value }) => ({ date, value }));
  if (Array.isArray(raw)) return take(raw).map(({ date, value }) => ({ date, value }));
  if (raw?.data && Array.isArray(raw.data)) return take(raw.data).map(({ date, value }) => ({ date, value }));
  return [];
}

// ====== 类型（统一声明一次）======
type HistoryPoint = { date: string; close: number };

// ====== 批量结果 → PredictionData 适配器 ======
function adaptRowToPredictionData(r: any, fallbackMethod: string): PredictionData {
  // 1) 后端很多字段在 r.result 里，这里把它拍平
  const inner = (r && typeof r === "object" && r.result && typeof r.result === "object")
    ? { ...r, ...r.result }
    : r || {};

  // 2) 基本元数据
  const symbol =
    inner.symbol || inner.ticker || r?.symbol || r?.ticker || "UNKNOWN";

  const companyName =
    inner.companyName || inner.company || inner.company_name || r?.company_name || symbol;

  const method =
    inner.method || r?.method || r?.meta?.method || fallbackMethod;

  const updatedRaw =
    inner.generated_at || inner.updated_at || inner.updatedAt ||
    inner.timestamp || inner.ts || inner.time || inner.as_of;

  // 3) 当前价格
  const curRaw =
    inner.currentPrice ?? inner.current_price ??
    inner.price ?? inner.last ?? inner.close ?? inner.px_last ?? inner.y ?? inner.value;
  const currentPrice = Number(curRaw);

  // 4) 预测数组：兼容多种字段名与嵌套
  let rawPreds: any[] = [];
  if (Array.isArray(inner.predictions)) rawPreds = inner.predictions;
  else if (Array.isArray(inner.forecast)) rawPreds = inner.forecast;
  else if (Array.isArray(inner.forecast_points)) rawPreds = inner.forecast_points;
  else if (Array.isArray(inner.points)) rawPreds = inner.points;
  else if (Array.isArray(inner.result)) rawPreds = inner.result;
  else if (Array.isArray(inner.preds)) rawPreds = inner.preds;
  else if (inner.horizons && typeof inner.horizons === "object") {
    rawPreds = Object.entries(inner.horizons).map(([h, v]) => ({ period: h, value: v }));
  }

  // 5) 统一字段名（特别是 horizon_days / predicted）
  const preds = (rawPreds || [])
    .map((p: any) => ({
      horizon:
        p.horizon ?? p.horizon_days ?? p.h ?? p.period ?? p.name ?? "",
      value: Number(p.value ?? p.predicted ?? p.price ?? p.yhat ?? p.y_pred ?? p.v ?? p[1]),
      confidence: Number(p.confidence ?? inner.confidence ?? r?.confidence ?? NaN),
    }))
    .filter((x) => Number.isFinite(x.value));

  function labelOf(h: any): string {
    const s = String(h).toLowerCase().trim();
    if (s === "1" || /(^|\s)1d(\s|$)|^1\s*day$/.test(s)) return "1 Day";
    if (s === "2" || /(^|\s)2d(\s|$)|^2\s*day$/.test(s)) return "2 Days";
    if (s === "3" || /(^|\s)3d(\s|$)|^3\s*day$/.test(s)) return "3 Days";
    if (s === "4" || /(^|\s)4d(\s|$)|^4\s*day$/.test(s)) return "4 Days";
    if (s === "5" || /(^|\s)5d(\s|$)|^5\s*day$/.test(s)) return "5 Days";
    if (s === "6" || /(^|\s)6d(\s|$)|^6\s*day$/.test(s)) return "6 Days";
    if (s === "7" || /(^|\s)7d(\s|$)|^7\s*day$|1\s*week/.test(s)) return "1 Week";
    if (s === "30" || /(^|\s)30d(\s|$)|^30\s*day$|1\s*month|^1m$/.test(s)) return "1 Month";
    return s || "Unknown";
  }

  const predictions =
    preds.length > 0
      ? preds.map((p) => ({
          period: labelOf(p.horizon),
          predictedPrice: Number(p.value),
          confidence: Number.isFinite(p.confidence) ? p.confidence : 0.7, // 单点置信度优先
          change: Number.isFinite(currentPrice) ? Number(p.value) - Number(currentPrice) : 0,
          changePercent:
            Number.isFinite(currentPrice) && Number(currentPrice) !== 0
              ? ((Number(p.value) - Number(currentPrice)) / Number(currentPrice)) * 100
              : 0,
        }))
      : [
          {
            period: "1 Day",
            predictedPrice: Number.isFinite(currentPrice) ? Number(currentPrice) : 0,
            confidence: 0.7,
            change: 0,
            changePercent: 0,
          },
        ];

  return {
    symbol,
    companyName,
    method,
    currentPrice: Number.isFinite(currentPrice) ? Number(currentPrice) : 0,
    accuracy: Number(inner.accuracy ?? r?.accuracy ?? 0.7),
    risk: (inner.risk_level || inner.risk || r?.risk_level || r?.risk || "Medium") as any,
    lastUpdated: updatedRaw ? String(updatedRaw) : new Date().toISOString(),
    predictions,
  } as PredictionData;
}
/* --------------------- 本地诊断算法（轻量） --------------------- */
type RiskLevel = "Low" | "Medium" | "High";
type DayBar = { date: string; close: number };
type ChartRow = { date: string; value?: number; isFuture?: boolean };

function directionalAccuracy(closes: number[], window = 90): number {
  if (closes.length < 3) return 0.7;                 // 样本太少给默认
  const diffs = [];
  for (let i = 1; i < closes.length; i++) diffs.push(closes[i] - closes[i-1]);
  const recent = diffs.slice(-Math.max(10, Math.min(window, diffs.length)));
  if (recent.length < 2) return 0.7;
  let hits = 0;
  for (let i = 1; i < recent.length; i++) {
    if ((recent[i-1] >= 0) === (recent[i] >= 0)) hits++;
  }
  return Math.round((hits / (recent.length - 1)) * 10000) / 10000;
}
function stdPopulation(xs: number[]): number {
  if (xs.length <= 1) return 0;
  const mu = xs.reduce((a,b)=>a+b,0)/xs.length;
  const varp = xs.reduce((s,x)=>s+(x-mu)*(x-mu),0)/xs.length;
  return Math.sqrt(varp);
}
function riskLevelByVol(closes: number[]): RiskLevel {
  if (closes.length < 20) return "Medium";
  const rets: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    const prev = closes[i-1];
    if (prev !== 0) rets.push(closes[i]/prev - 1);
  }
  const vol = stdPopulation(rets);
  if (vol < 0.012) return "Low";
  if (vol < 0.025) return "Medium";
  return "High";
}
function bestTimeframeFromPredictions(preds: PredictionData["predictions"]): string {
  if (!preds?.length) return "1 Day";
  const best = [...preds].sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))[0];
  return best?.period || "1 Day";
}


// ====== 组件 ======
const METHOD_OPTIONS: { value: string; label: string }[] = [
  { value: "lstm", label: "LSTM" },
  { value: "arima", label: "ARIMA" },
  { value: "prophet", label: "Prophet" },
  { value: "lgbm", label: "LightGBM" },
  { value: "seq2seq", label: "Seq2Seq" },
  { value: "transformer", label: "Transformer" },
  { value: "ma", label: "MA" },
];

export const PredictTab: React.FC = () => {
  const { user } = useAuth();

  const [list, setList] = useState<PredictionData[]>([]);
  const [symbols, setSymbols] = useState<string[]>(DEFAULT_SYMBOLS);

  const [selectedSymbol, setSelectedSymbol] = useState<string>(DEFAULT_SYMBOLS[0]);
  const [selectedStock, setSelectedStock] = useState<PredictionData | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<string>("1 Day");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [method, setMethod] = useState<string>("transformer");
  const [limit, setLimit] = useState<number>(10);
  const [hist7Map, setHist7Map] = useState<Record<string, HistoryPoint[]>>({});

  const [chartErr, setChartErr] = useState<string | null>(null);
  const [history7, setHistory7] = useState<HistoryPoint[]>([]);
  const [forecastRes, setForecastRes] = useState<any | null>(null);

  const symbol = selectedSymbol || "AAPL";
  const HORIZONS = [1, 2, 3, 4, 5, 6, 7, 30];
  const cacheKey = useMemo(() => `${method}|${limit}|${HORIZONS.join(",")}|${symbols.join(",")}`, [method, limit, symbols]);
  const [refreshNonce, setRefreshNonce] = useState(0);

  const [diag, setDiag] = useState<{
  accuracy?: number;
  risk_level?: "Low" | "Medium" | "High" | string;
  best_timeframe?: string;
  data_points?: number;
  generated_at?: string;
} | null>(null);

  const companyName = useMemo(() => {
    const idx = symbols.indexOf(selectedSymbol);
    return companyNames[idx] ?? selectedSymbol;
  }, [selectedSymbol, symbols]);

  const derivedDiag = useMemo(() => {
    const closes = (history7 || []).map((d) => Number(d.close)).filter((x) => Number.isFinite(x));
    const accuracy = directionalAccuracy(closes, 90);
    const risk_level = riskLevelByVol(closes);
    const best_timeframe = bestTimeframeFromPredictions(selectedStock?.predictions || []);
    const data_points =
      Number(forecastRes?.meta?.n_obs) ||
      Number(forecastRes?.n_obs) ||
      closes.length;
    return {
      accuracy,
      risk_level,
      best_timeframe,
      data_points,
      generated_at: new Date().toISOString(),
    };
  }, [history7, selectedStock?.predictions, forecastRes?.meta?.n_obs, forecastRes?.n_obs]);

  
// 任一关键UI状态变化 -> 持久化
useEffect(() => {
  const state: UIState = { symbol: selectedSymbol, method, limit };
  ssWrite(UI_STATE_KEY, state);
}, [selectedSymbol, method, limit]);


useEffect(() => {
  let alive = true;
  (async () => {
    try {
      // 可选：在 Network 里能看到
      // console.log("[diag] fetch", symbol, method);
      const d = await fetchForecastDiagnostics(symbol, method);
      if (!alive) return;
      const next = {
        accuracy: Number(d?.accuracy),
        risk_level: (d?.risk_level as any) || "Medium",
        best_timeframe: d?.best_timeframe || undefined,
        data_points: Number(d?.data_points),
        generated_at: String(d?.generated_at || ""),
      };
      setDiag(next);
      // 如需缓存再写入；别在请求前 short-circuit
      // sessionStorage.setItem(`diag|${symbol}|${method}`, JSON.stringify({ at: Date.now(), data: next }));
    } catch (e) {
      if (!alive) return;
      setDiag(null); // 失败则留空
    }
  })();
  return () => { alive = false; };
}, [symbol, method]);
  // ====== 加载推荐 + 批量预测 ======
  // ====== 加载推荐 + 批量预测（带缓存）======
useEffect(() => {
  let alive = true;
  (async () => {
    setLoading(true);
    setErr(null);

    try {
      // 0) 推荐 symbols
      const uid = user?.id || "demo";
      let recTickers: string[] = [];
      try {
        recTickers = await fetchRecommendedSymbols(uid, limit);
      } catch {
        recTickers = [];
      }
      if (!alive) return;

      const targetSymbols = (recTickers.length ? recTickers : DEFAULT_SYMBOLS).slice(0, limit);
      setSymbols(targetSymbols);

      // 1) 先读批量结果缓存
      const cached = batchCache.get(cacheKey);
      if (cached && isFresh(cached.at)) {
        const items = cached.data;
        setList(items);

        // 恢复 UI 选择
        const ui = ssRead<UIState>(UI_STATE_KEY);
        const sym0 = ui?.symbol && targetSymbols.includes(ui.symbol) ? ui.symbol : (items[0]?.symbol ?? targetSymbols[0] ?? "AAPL");
        setSelectedSymbol(sym0);
        const first = items.find(x => x.symbol === sym0) || items[0] || null;
        setSelectedStock(first ?? null);
        if (first) {
          const day =
            first.predictions.find((p) => /1\s*day/i.test(p.period))?.period ??
            first.predictions[0]?.period ?? "1 Day";
          setSelectedPeriod(day);
        }
        return; // 命中缓存，直接返回
      }

      // 2) 未命中缓存 -> 请求
      const raw = await fetchForecastBatch({
        method,
        horizons: HORIZONS,
        limit,
        symbols: targetSymbols,
      });

      const rows = normalizeBatchPayload(raw);
      if (!rows.length) throw new Error("Empty or invalid batch payload");

      const items: PredictionData[] = rows.map((r: any) => {
        const mapped = adaptRowToPredictionData(r, method);
        return {
          ...mapped,
          lastUpdated: relativeFromNow(mapped.lastUpdated as unknown as string),
        };
      });

      const filtered = items.filter((x) => targetSymbols.includes(x.symbol));
      setList(filtered);

      // 写入缓存（内存 + 会话）
      batchCache.set(cacheKey, { at: Date.now(), data: filtered });
      persistCacheToSession();

      // 默认/恢复选择
      const ui = ssRead<UIState>(UI_STATE_KEY);
      const sym0 = ui?.symbol && targetSymbols.includes(ui.symbol) ? ui.symbol : (filtered[0]?.symbol ?? targetSymbols[0] ?? "AAPL");
      setSelectedSymbol(sym0);
      const first = filtered.find(x => x.symbol === sym0) || filtered[0] || null;
      setSelectedStock(first ?? null);
      if (first) {
        const day =
          first.predictions.find((p) => /1\s*day/i.test(p.period))?.period ??
          first.predictions[0]?.period ?? "1 Day";
        setSelectedPeriod(day);
      }
    } catch (e: any) {
      if (alive) setErr(e?.message || "Load failed");
    } finally {
      if (alive) setLoading(false);
    }
  })();
  return () => { alive = false; };
}, [user?.id, method, limit, refreshNonce]);
useEffect(() => {
  let alive = true;
  (async () => {
    try {
      const cacheKey = `diag|${symbol}|${method}`;
      const cached = ssRead<typeof diag>(cacheKey);
      if (cached) { setDiag(cached); return; }

      const d = await fetchForecastDiagnostics(symbol, method);
      if (!alive) return;
      const next = {
        accuracy: Number(d?.accuracy),
        risk_level: (d?.risk_level as any) || "Medium",
        best_timeframe: d?.best_timeframe || undefined,
        data_points: Number(d?.data_points),
        generated_at: String(d?.generated_at || ""),
      };
      setDiag(next);
      ssWrite(cacheKey, next);
    } catch {
      if (!alive) return;
      setDiag(null);
    }
  })();
  return () => { alive = false; };
}, [symbol, method]);

  // ====== 单只股票的历史7天与未来7天 ======
  // ====== 单只股票的历史7天与未来7天（带缓存）======
useEffect(() => {
  let alive = true;
  (async () => {
    try {
      setChartErr(null);
      const histKey = `h7|${symbol}`;
      const fcKey   = `fc7|${symbol}`;

      // 先尝试读缓存
      const histCached = ssRead<HistoryPoint[]>(histKey);
      const fcCached   = ssRead<any>(fcKey);

      if (histCached && fcCached) {
        if (!alive) return;
        setHistory7(histCached);
        setForecastRes(fcCached);
        return;
      }

      // 未命中 -> 请求（单票接口也加一层请求缓存）
      const [h, f] = await Promise.all([
        getJSONCached(`${API}/forecast/prices7?ticker=${encodeURIComponent(symbol)}`),
        getJSONCached(`${API}/forecast/${encodeURIComponent(symbol)}?horizons=1,2,3,4,5,6,7`)
      ]);
      if (!alive) return;

      // const todayISO = new Date().toISOString().slice(0, 10);
      // const hist: DayBar[] = (Array.isArray(h?.prices) ? h.prices : [])
      //   .map((p: any) => ({ date: String(p.date).slice(0, 10), close: Number(p.close) }))
      //   .filter((p: DayBar) => p.date && p.date < todayISO && Number.isFinite(p.close))
      //   .sort((a: DayBar, b: DayBar) => a.date.localeCompare(b.date))

const hist: DayBar[] = (Array.isArray(h?.prices) ? h.prices : [])
  .map((p: any) => ({ date: String(p.date).slice(0, 10), close: Number(p.close) }))
  .sort((a: DayBar, b: DayBar) => a.date.localeCompare(b.date));
// 这里不要再 filter(< todayISO)，也不要再 slice(-7)


      setHistory7(hist);
      setForecastRes(f);

      // 落缓存（会话）
      ssWrite(histKey, hist);
      ssWrite(fcKey, f);
    } catch (e: any) {
      if (!alive) return;
      setChartErr(e?.message ?? "Load error");
    }
  })();
  return () => { alive = false; };
}, [symbol]);

// 初次挂载尝试恢复 UI 状态（方法/limit/符号）
useEffect(() => {
  const ui = ssRead<UIState>(UI_STATE_KEY);
  if (ui) {
    setMethod(ui.method || "transformer");
    setLimit(Number.isFinite(ui.limit) ? ui.limit : 10);
    // selectedSymbol 会在批量加载后做最终校验/恢复
    if (ui.symbol) setSelectedSymbol(ui.symbol);
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, []);

// ✅ 直接可用的数据
const currentPriceSafe = Number.isFinite(selectedStock?.currentPrice)
  ? Number(selectedStock!.currentPrice) : NaN;

const chosenPred =
  selectedStock?.predictions.find((p) => p.period === selectedPeriod) ||
  selectedStock?.predictions[0] ||
  null;

const chosenPredPrice = Number.isFinite(chosenPred?.predictedPrice as any)
  ? Number(chosenPred!.predictedPrice)
  : NaN;

const chosenPredConfPct = Math.round((chosenPred?.confidence ?? 0) * 100);

// 数据点：优先后端 meta → 其次 n_obs → 再退化到已有历史条数
const dataPointsDirect = Number(
  (forecastRes?.meta?.n_obs ?? forecastRes?.n_obs ?? history7?.length ?? 0)
);

// 更新时间：优先 selectedStock.lastUpdated（你前面已转成 “xx minutes ago”）
const updatedTextDirect = selectedStock?.lastUpdated
  ? `Updated ${selectedStock.lastUpdated}`
  : "—";

// 右上角“最佳周期”直接显示当前选择的周期；如果想显示第一个也可：
const bestTimeframeDirect = selectedPeriod || chosenPred?.period || "—";

// 风险：直接用批量结果里的 risk（已经适配过）
const riskDirect = (selectedStock?.risk as any) || "—";


const accPct = Number.isFinite(diag?.accuracy as any)
  ? Math.round(Number(diag!.accuracy) * 100)
  : Math.round(Number(selectedStock?.accuracy ?? 0) * 100);

const risk = (diag?.risk_level as any)
  || (selectedStock?.risk as any)
  || "Medium";

const bestLabel = diag?.best_timeframe
  || (selectedStock?.predictions.length
        ? [...selectedStock!.predictions].sort(
            (a, b) => (b.confidence ?? 0) - (a.confidence ?? 0)
          )[0]?.period ?? "1 Day"
        : "1 Day");

const dp = Number.isFinite(diag?.data_points as any)
  ? Number(diag!.data_points)
  : (
      Number(forecastRes?.meta?.n_obs) ||
      Number(forecastRes?.n_obs) ||
      0
    );

const updatedText = diag?.generated_at
  ? `Updated ${relativeFromNow(diag.generated_at)}`
  : (selectedStock?.lastUpdated ? `Updated ${selectedStock.lastUpdated}` : "—");


  const currentPrediction =
    selectedStock?.predictions.find((p) => p.period === selectedPeriod) ||
    selectedStock?.predictions[0] || {
      period: "1 Day",
      predictedPrice: selectedStock?.currentPrice ?? 0,
      confidence: 0.7,
      change: 0,
      changePercent: 0,
    };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "Low":
        return "text-green-600 bg-green-50 border-green-200";
      case "Medium":
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "High":
        return "text-red-600 bg-red-50 border-red-200";
      default:
        return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  // ========== 图表数据（历史→今天→未来7天；统一 value 字段） ==========
// ========== 图表数据（历史→今天→未来7天；统一用右侧 predictions） ==========
const chartData = useMemo<ChartRow[]>(() => {
  // 历史段
  const histRows: ChartRow[] = Array.isArray(history7)
    ? history7.map((h) => ({ date: String(h.date).slice(0, 10), value: Number(h.close), isFuture: false }))
    : [];

  const todayISO = new Date().toISOString().slice(0, 10);
  const lastHistValue = histRows.length > 0 ? Number(histRows[histRows.length - 1].value) : NaN;

  // 当前价：优先 selectedStock.currentPrice，退化到最近历史价
  const curPrice = Number.isFinite(selectedStock?.currentPrice)
    ? Number(selectedStock!.currentPrice)
    : lastHistValue;

  // —— 1) 用右侧 predictions 构造“锚点” —— //
  const periodToDays = (s: string) => {
    const t = String(s || "").trim().toLowerCase();
    if (/^1\s*day$|^1d$/.test(t)) return 1;
    if (/^2\s*days$|^2d$/.test(t)) return 2;
    if (/^3\s*days$|^3d$/.test(t)) return 3;
    if (/^4\s*days$|^4d$/.test(t)) return 4;
    if (/^5\s*days$|^5d$/.test(t)) return 5;
    if (/^6\s*days$|^6d$/.test(t)) return 6;
    if (/^1\s*week$|^7d$|^7\s*days$/.test(t)) return 7;
    if (/^1\s*month$|^30d$/.test(t)) return 30;
    // 兜底：抽取数字
    const n = parseInt(t.replace(/[^\d]/g, ""), 10);
    return Number.isFinite(n) && n > 0 ? n : NaN;
  };

  const rightPreds = Array.isArray(selectedStock?.predictions) ? selectedStock!.predictions : [];
  const anchorsFromRight = rightPreds
    .map((p) => ({ days: periodToDays(p.period), value: Number(p.predictedPrice) }))
    .filter((a) => Number.isFinite(a.days) && Number.isFinite(a.value))
    .sort((a, b) => a.days - b.days);

  // 目标日期 D+1 ~ D+7
  const D = new Date(todayISO);
  const futureDates: string[] = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(D);
    d.setDate(d.getDate() + (i + 1));
    return d.toISOString().slice(0, 10);
  });

  // S 曲线（平滑插值）
  const easeCos = (t: number) => (1 - Math.cos(Math.PI * t)) / 2;

  // —— 2) 生成未来逐日（优先用右侧锚点；没有则回退 forecastRes） —— //
  let futureDaily: Array<{ date: string; value: number }> = [];

  if (anchorsFromRight.length && Number.isFinite(curPrice)) {
    // 以今天作为 day=0 的锚
    const anchor0 = { days: 0, value: Number(curPrice) };
    const anchors = [anchor0, ...anchorsFromRight]
      .filter((x, idx, arr) => arr.findIndex((y) => y.days === x.days) === idx)
      .sort((a, b) => a.days - b.days);

    futureDaily = futureDates.map((dt, i) => {
      const day = i + 1; // 1..7
      // 找到右侧第一个 >= day 的锚点
      const rIdx = anchors.findIndex((a) => a.days >= day);
      if (rIdx === -1) {
        // 超出最大锚：取最后一个锚点的值（保持平）
        return { date: dt, value: anchors[anchors.length - 1].value };
        }
      if (rIdx === 0) {
        // 还没到第一个锚：用第一个锚与 day=0 插值
        const L = anchors[0]; // 其实就是 day=0
        const R = anchors[1] || anchors[0];
        const span = Math.max(1, R.days - L.days);
        const t = (day - L.days) / span;
        return { date: dt, value: Number(L.value + (R.value - L.value) * easeCos(t)) };
      }
      const L = anchors[rIdx - 1];
      const R = anchors[rIdx];
      if (R.days === L.days) return { date: dt, value: R.value };
      const span = Math.max(1, R.days - L.days);
      const t = (day - L.days) / span;
      return { date: dt, value: Number(L.value + (R.value - L.value) * easeCos(t)) };
    });

  } else {
    // 回退：用单票接口的 forecastRes（旧逻辑）
    const rawPreds = normalizeForecast(forecastRes)
      .map((p) => ({ date: String(p.date).slice(0, 10), value: Number(p.value) }))
      .filter((p) => p.date && Number.isFinite(p.value));

    if (rawPreds.length && Number.isFinite(curPrice)) {
      const anchors = [{ date: todayISO, value: Number(curPrice) }, ...rawPreds]
        .filter((x, i, arr) => arr.findIndex((y) => y.date === x.date) === i)
        .sort((a, b) => a.date.localeCompare(b.date));

      futureDaily = futureDates.map((dt) => {
        const rIdx = anchors.findIndex((a) => a.date >= dt);
        if (rIdx <= 0) return { date: dt, value: anchors[0].value };
        const L = anchors[rIdx - 1];
        const R = anchors[rIdx];
        if (L.date === R.date) return { date: dt, value: R.value };
        const t = (new Date(dt).getTime() - new Date(L.date).getTime()) /
                  (new Date(R.date).getTime() - new Date(L.date).getTime());
        return { date: dt, value: Number(L.value + (R.value - L.value) * easeCos(t)) };
      });
    }
  }

  // 合并（历史 + 今天 + 未来）
  const map = new Map<string, ChartRow>();
  for (const r of histRows) map.set(r.date, { ...r, isFuture: false });
  if (Number.isFinite(curPrice)) map.set(todayISO, { date: todayISO, value: Number(curPrice), isFuture: false });
  for (const p of futureDaily) map.set(p.date, { date: p.date, value: Number(p.value), isFuture: true });

  return Array.from(map.values())
    .filter((x) => Number.isFinite(x.value))
    .sort((a, b) => a.date.localeCompare(b.date));
}, [history7, forecastRes, selectedStock?.currentPrice, selectedStock?.predictions]);

  const fmtMoney = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
  const todayISO = new Date().toISOString().slice(0, 10);
  const formatDate = (dateStr: string) => {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    return `${(d.getMonth() + 1).toString().padStart(2, "0")}-${d.getDate().toString().padStart(2, "0")}`;
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h2
  className="text-4xl font-extrabold bg-gradient-to-r from-blue-600 via-indigo-500 to-cyan-400 
             bg-clip-text text-transparent drop-shadow-sm tracking-tight"
>
  Stock Trend Predictions
</h2>
          {/* <p className="text-gray-600">AI-powered price forecasting with confidence intervals</p> */}
        </div>

        {/* Controls */}
{/* Controls – 居中版（Tickers / Methods / Limit） */}
{/* Controls – 蓝色渐变美化版 */}
<div className="flex flex-wrap items-end gap-6 justify-center">

  {/* Tickers */}
  <div className="flex flex-col items-center text-center">
    <label className="text-sm font-semibold text-gray-700 mb-2">Tickers</label>
    <div className="relative">
      <select
        value={selectedSymbol}
        onChange={(e) => {
          const sym = e.target.value;
          setSelectedSymbol(sym);
          const match = list.find((x) => x.symbol === sym);
          if (match) setSelectedStock(match);
        }}
        className="appearance-none pl-5 pr-10 py-3 text-base font-semibold rounded-2xl 
                   border border-blue-200 bg-gradient-to-r from-blue-50 to-cyan-50 
                   shadow-sm hover:shadow-md text-center text-blue-800 
                   focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 
                   transition-all min-w-[150px]"
      >
        {(list.length ? list.map((x) => x.symbol) : symbols).map((sym) => (
          <option key={sym} value={sym}>{sym}</option>
        ))}
      </select>
      <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-blue-500 text-lg">▾</span>
    </div>
  </div>

  {/* Methods */}
  <div className="flex flex-col items-center text-center">
    <label className="text-sm font-semibold text-gray-700 mb-2">Methods</label>
    <div className="relative">
      <select
        value={method}
        onChange={(e) => setMethod(e.target.value)}
        className="appearance-none pl-5 pr-10 py-3 text-base font-semibold rounded-2xl 
                   border border-blue-200 bg-gradient-to-r from-blue-50 to-cyan-50 
                   shadow-sm hover:shadow-md text-center text-blue-800 
                   focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 
                   transition-all min-w-[150px]"
      >
        {METHOD_OPTIONS.map((m) => (
          <option key={m.value} value={m.value}>{m.label}</option>
        ))}
      </select>
      <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-blue-500 text-lg">▾</span>
    </div>
  </div>

  {/* Limit */}
  <div className="flex flex-col items-center text-center">
    <label className="text-sm font-semibold text-gray-700 mb-2">Limit</label>
    <div className="flex items-center justify-center px-4 py-2.5 rounded-2xl 
                    border border-blue-200 bg-gradient-to-r from-blue-50 to-cyan-50 
                    shadow-sm hover:shadow-md transition-all">
      <input
        type="number"
        inputMode="numeric"
        min={1}
        step={1}
        value={limit}
        onChange={(e) => {
          const v = parseInt(e.target.value || "1", 10);
          setLimit(Number.isFinite(v) ? Math.max(1, v) : 1);
        }}
        className="w-24 px-3 py-2 text-base font-semibold text-center text-blue-800 rounded-xl 
                   border border-blue-200 bg-white focus:outline-none 
                   focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition-all"
      />
    </div>
  </div>

  {/* Refresh */}
  <div className="flex flex-col items-center text-center">
    <label className="text-sm font-semibold text-gray-700 mb-2 invisible select-none">Refresh</label>
    <button
      onClick={() => {
        batchCache.delete(cacheKey);
        setRefreshNonce((n) => n + 1);
          sessionStorage.removeItem(`diag|${selectedSymbol}|${method}`); // 清诊断缓存
  setLimit((l) => l); // 或者 setRefreshNonce(n => n + 1)
      }}
      disabled={loading}
      className={[
        "inline-flex items-center justify-center gap-3 px-6 py-3 rounded-2xl text-base font-semibold",
        "border border-blue-200 bg-gradient-to-r from-blue-100 to-cyan-100 text-blue-800 shadow-sm",
        "hover:from-blue-200 hover:to-cyan-200 hover:shadow-md",
        "transition-all active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400",
        loading ? "opacity-60 cursor-not-allowed" : ""
      ].join(" ")}
    >
      {loading ? (
        <>
          <Loader2 className="w-5 h-5 animate-spin" />
          Refreshing...
        </>
      ) : (
        <>
          <RefreshCcw className="w-5 h-5" />
          Refresh
        </>
      )}
    </button>
  </div>
</div>


      </div>

      {/* 错误与加载 */}
      {err && <div className="p-3 rounded bg-red-50 border border-red-200 text-red-700">{err}</div>}
      {loading && <div className="text-sm text-gray-500">Loading...</div>}

      {/* 内容 */}
      {!selectedStock ? null : (
        <>
          {/* Model Performance */}
{/* Model Performance – 数据驱动版（直接替换原来的四宫格） */}
{/* Model Performance – 前端诊断版 */}
{/* ==== 四宫格：直接数据显示 ==== */}
<div className="grid grid-cols-1 md:grid-cols-4 gap-4">
  {/* 1️⃣ 当前价格 */}
  <div className="rounded-2xl border border-blue-200 bg-gradient-to-r from-blue-50 to-white p-5">
    <div className="flex items-center gap-2 mb-1">
      <Target className="h-5 w-5 text-blue-600" />
      <span className="text-sm font-medium text-blue-700">Current Price</span>
    </div>
    <p className="text-3xl font-extrabold text-blue-700 leading-tight">
      {Number.isFinite(selectedStock?.currentPrice)
        ? `$${selectedStock.currentPrice.toFixed(2)}`
        : "—"}
    </p>
    <p className="text-xs text-blue-600 mt-0.5">
      {selectedStock?.lastUpdated
        ? `Updated ${selectedStock.lastUpdated}`
        : "—"}
    </p>
  </div>

  {/* 2️⃣ 选中周期预测价格 */}
  <div className="rounded-2xl border border-green-200 bg-gradient-to-r from-green-50 to-white p-5">
    <div className="flex items-center gap-2 mb-1">
      <Calendar className="h-5 w-5 text-green-700" />
      <span className="text-sm font-medium text-green-700">Forecast Price ({selectedPeriod})</span>
    </div>
    <p className="text-3xl font-extrabold text-green-800 leading-tight">
      {Number.isFinite(currentPrediction?.predictedPrice)
        ? `$${currentPrediction.predictedPrice.toFixed(2)}`
        : "—"}
    </p>
    <p className="text-xs text-green-700 mt-0.5">
      Based on model: {selectedStock?.method?.toUpperCase()}
    </p>
  </div>

  {/* 3️⃣ 置信度 */}
  <div className="rounded-2xl border border-amber-200 bg-gradient-to-r from-amber-50 to-white p-5">
    <div className="flex items-center gap-2 mb-1">
      <AlertCircle className="h-5 w-5 text-amber-700" />
      <span className="text-sm font-medium text-amber-700">Confidence</span>
    </div>
    <p className="text-3xl font-extrabold text-amber-800 leading-tight">
      {Number.isFinite(currentPrediction?.confidence)
        ? `${Math.round(currentPrediction.confidence * 100)}%`
        : "—"}
    </p>
    <p className="text-xs text-amber-700 mt-0.5">
      Confidence for {selectedPeriod} prediction
    </p>
  </div>

  {/* 4️⃣ 涨跌幅 */}
  <div className="rounded-2xl border border-purple-200 bg-gradient-to-r from-purple-50 to-white p-5">
    <div className="flex items-center gap-2 mb-1">
      <BarChart className="h-5 w-5 text-purple-700" />
      <span className="text-sm font-medium text-purple-700">Change %</span>
    </div>
    {(() => {
      const cur = Number(selectedStock?.currentPrice);
      const pred = Number(currentPrediction?.predictedPrice);
      if (!Number.isFinite(cur) || !Number.isFinite(pred) || cur === 0)
        return <p className="text-3xl font-extrabold text-gray-500 leading-tight">—</p>;
      const delta = pred - cur;
      const pct = (delta / cur) * 100;
      const up = delta >= 0;
      return (
        <>
          <p
            className={`text-3xl font-extrabold leading-tight ${
              up ? "text-green-700" : "text-red-700"
            }`}
          >
            {up ? "▲" : "▼"} {pct.toFixed(2)}%
          </p>
          <p className="text-xs text-purple-700 mt-0.5">
            {up ? "Expected increase" : "Expected decrease"}
          </p>
        </>
      );
    })()}
  </div>
</div>




          {/* Stock Info + Chart */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center">
                  <span className="text-lg font-bold text-gray-700">{selectedStock.symbol.charAt(0)}</span>
                </div>
                <div>
<h3 className="text-2xl font-extrabold text-gray-900 tracking-tight">{selectedStock.symbol}</h3>
<span className="px-3 py-1 text-sm rounded-full border border-blue-200 bg-blue-50 text-blue-700 font-medium">
  Method: {selectedStock.method}
</span>
                  {/* <p className="text-gray-600">{selectedStock.companyName}</p> */}
                </div>
              </div>
{/* 右侧：显示选中周期的预测价 */}
<div className="text-right">
  {(() => {
    const pred = Number(currentPrediction?.predictedPrice);
    const cur  = Number(selectedStock?.currentPrice);
    const hasPred = Number.isFinite(pred);
    const hasCur  = Number.isFinite(cur);
    const delta   = hasPred && hasCur ? pred - cur : NaN;
    const pct     = hasPred && hasCur && cur !== 0 ? (delta / cur) * 100 : NaN;
    const up      = Number.isFinite(delta) && delta >= 0;

    return (
      <>
<p className="text-4xl font-extrabold text-gray-900 leading-tight">
  {fmtMoney.format(currentPrediction.predictedPrice ?? 0)}
</p>
<p className="text-sm text-gray-600">{selectedPeriod} Forecast</p>

        {hasCur && hasPred ? (
          <p className={`text-xs mt-1 ${up ? "text-green-600" : "text-red-600"}`}>
            {up ? "▲" : "▼"} {fmtMoney.format(Math.abs(delta))} ({Math.abs(pct).toFixed(2)}%)
            <span className="text-gray-500"> vs Current</span>
          </p>
        ) : null}
      </>
    );
  })()}
</div>

            </div>

            {/* Periods */}
            <div className="flex space-x-2 mb-6">
              {selectedStock.predictions.map((pred) => (
                <button
                  key={pred.period}
                  onClick={() => setSelectedPeriod(pred.period)}
                  className={`px-5 py-2.5 rounded-lg text-base font-semibold transition-colors ${
                    selectedPeriod === pred.period ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  {pred.period}
                </button>
              ))}
            </div>

            {/* Chart */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
<div className="space-y-4">
  <div className="rounded-2xl border border-gray-300 p-5 shadow-md bg-white">
    <div className="flex items-center gap-2 mb-4">
      <Calendar className="w-5 h-5 text-blue-600" />
<span className="text-lg font-semibold text-gray-800">Stock Trend Chart</span>

    </div>

    <div className="rounded-2xl border border-gray-200 shadow-md p-5 bg-gradient-to-b from-white to-gray-50">
      <div style={{ width: "100%", height: 460 }}>
        <ResponsiveContainer>
          <LineChart data={chartData} margin={{ left: 20, right: 25, top: 12, bottom: 12 }}>
            <defs>
              {/* 更深的线条渐变 */}
              <linearGradient id="priceStroke" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#1d4ed8" /> {/* 深蓝 */}
                <stop offset="100%" stopColor="#5b21b6" /> {/* 深紫 */}
              </linearGradient>
              <linearGradient id="historyFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="rgba(29,78,216,0.25)" />
                <stop offset="100%" stopColor="rgba(29,78,216,0.05)" />
              </linearGradient>
              <linearGradient id="futureFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="rgba(91,33,182,0.25)" />
                <stop offset="100%" stopColor="rgba(91,33,182,0.05)" />
              </linearGradient>
            </defs>

            {/* 背景区域填充 */}
            {chartData?.length ? (
              <ReferenceArea x1={chartData[0].date} x2={todayISO} fill="url(#historyFill)" strokeOpacity={0} />
            ) : null}
            {chartData?.length ? (
              <ReferenceArea
                x1={todayISO}
                x2={chartData[chartData.length - 1].date}
                fill="url(#futureFill)"
                strokeOpacity={0}
              />
            ) : null}

            {/* 网格与坐标 */}
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(156,163,175,0.5)" />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              tick={{ fontSize: 13, fill: "#374151", fontWeight: 500 }}
              stroke="#d1d5db"
              axisLine={false}
              tickLine={false}
              minTickGap={28}
            />
            <YAxis
              type="number"
              dataKey="value"
              domain={[
                (dataMin: number) => Math.floor((dataMin ?? 0) * 0.98),
                (dataMax: number) => Math.ceil((dataMax ?? 0) * 1.02),
              ]}
              tickFormatter={(v: number) =>
                Number(v).toLocaleString("en-US", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })
              }
              tick={{ fontSize: 13, fill: "#374151", fontWeight: 500 }}
              axisLine={false}
              tickLine={false}
              width={80}
            />

            {/* Tooltip */}
            <Tooltip
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                const v = payload[0].payload?.value;
                const future = payload[0].payload?.isFuture;
                const color = future ? "#5b21b6" : "#1d4ed8";
                return (
                  <div className="rounded-xl border border-gray-300 bg-white/95 backdrop-blur p-3 shadow-2xl">
                    <div className="text-xs text-gray-500 mb-1">Date: {formatDate(label as string)}</div>
                    <div className="text-base font-semibold" style={{ color }}>

                      {fmtMoney.format(v ?? 0)}
                    </div>
                    <div className="text-[11px] text-gray-500 mt-0.5">
                      {future ? "Forecast" : "Historical / Today"}
                    </div>
                  </div>
                );
              }}
              cursor={{ stroke: "#a5b4fc", strokeWidth: 1 }}
            />

            {/* 今日参考线 */}
            <ReferenceLine
              x={todayISO}
              stroke="#6b7280"
              strokeDasharray="3 3"
              label={{
                value: "Today",
                position: "top",
                fill: "#4b5563",
                fontSize: 13,
                fontWeight: 500,
              }}
            />

            {/* 历史线 + 预测线 */}
            <Line
              type="monotone"
              dataKey="value"
              stroke="url(#priceStroke)"
              strokeWidth={3}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
            {/* <Line
              type="monotone"
              dataKey={(d: any) => (d.isFuture ? d.value : null)}
              stroke="#5b21b6"
              strokeWidth={2.5}
              strokeDasharray="5 3"
              dot={{ r: 4 }}
              isAnimationActive={false}
              connectNulls
            /> */}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>

    {chartErr && <div className="mt-2 text-sm text-red-600 text-center">{chartErr}</div>}
  </div>
</div>


              {/* Predictions list */}
{/* ====== All Predictions (Enhanced UI) ====== */}
<div className="space-y-4">
  <div className="rounded-2xl border border-blue-100 bg-gradient-to-b from-white via-blue-50 to-blue-100/30 shadow-md p-6">
    <div className="flex items-center justify-between mb-5">
      <h4 className="text-2xl font-extrabold text-blue-800 tracking-tight flex items-center gap-2">
        📊 All Predictions
      </h4>
      <span className="text-sm text-blue-600 font-medium">
        {selectedStock.symbol} • {selectedStock.method.toUpperCase()}
      </span>
    </div>

    {selectedStock.predictions.length ? (
      <div className="divide-y divide-blue-100">
        {selectedStock.predictions.map((pred) => {
          const up = pred.change >= 0;
          const color = up ? "text-green-600" : "text-red-600";
          const barColor = up
            ? "bg-gradient-to-r from-green-400 to-emerald-500"
            : "bg-gradient-to-r from-red-400 to-pink-500";

          return (
            <div
              key={pred.period}
              className="flex items-center justify-between py-3 hover:bg-white/70 transition-all duration-200 rounded-lg px-2"
            >
              {/* Period */}
              <div className="flex items-center gap-3">
                <div
                  className={`w-2.5 h-2.5 rounded-full ${
                    up ? "bg-green-500" : "bg-red-500"
                  }`}
                ></div>
                <span className="text-base font-medium text-gray-700">
                  {pred.period}
                </span>
              </div>

              {/* Value + Confidence */}
              <div className="text-right min-w-[130px]">
                <div className={`text-lg font-bold ${color}`}>
                  {Number.isFinite(pred.predictedPrice)
                    ? `$${pred.predictedPrice.toFixed(2)}`
                    : "—"}{" "}
                  {up ? "▲" : "▼"}
                </div>
                <div className="flex items-center justify-end gap-2 mt-1">
                  <div className="relative w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`${barColor} h-2 rounded-full transition-all`}
                      style={{ width: `${Math.round(pred.confidence * 100)}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500">
                    {Math.round(pred.confidence * 100)}%
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    ) : (
      <div className="text-sm text-gray-500 text-center py-6">
        No predictions yet.
      </div>
    )}
  </div>
</div>

            </div>
          </div>
        </>
      )}
    </div>
  );
};

