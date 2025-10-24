import React, { useEffect, useMemo, useState } from "react";
import { TrendingUp, TrendingDown, Calendar, Target, AlertCircle, BarChart } from "lucide-react";
import { fetchForecastBatch } from "../../api/forecast";
import { toPredictionData, type PredictionData } from "../../utils/forecastMapper";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { RefreshCcw, Loader2, BadgeInfo } from "lucide-react"; 


const symbols = ["AAPL", "TSLA", "NVDA"];
const companyNames = ["Apple Inc.", "Tesla Inc.", "NVIDIA Corp.", "Microsoft Corp."];

function relativeFromNow(iso: string) {
  const sec = Math.max(1, Math.floor((Date.now() - new Date(iso).getTime()) / 1000));
  if (sec < 60) return `${sec} seconds ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} minutes ago`;
  const hr = Math.floor(min / 60);
  return `${hr} hours ago`;
}

// 最近7天收盘价
const API = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function fetchLast7Prices(symbol: string) {
  const res = await fetch(`${API}/forecast/prices7?ticker=${encodeURIComponent(symbol)}`);
  if (!res.ok) throw new Error("Failed to load last 7 prices");
  return res.json(); // { prices: Array<{ date: string; close: number }> }
}

// 未来7天预测 —— 如果你的后端只支持 horizons=7，就把 URL 改成 ?horizons=7
async function fetchForecast7(symbol: string) {
  const res = await fetch(`${API}/forecast/${encodeURIComponent(symbol)}?horizon_days=7`);
  if (!res.ok) throw new Error("Failed to load 7-day forecast");
  return res.json(); // 兼容 ForecastResult { points?: [{date,value,type}] }
}

// cache
type BatchKey = string;

const CACHE_TTL_MS = 5 * 60 * 1000; // 5分钟，你可调
const batchCache = new Map<BatchKey, { at: number; data: PredictionData[] }>();
const inFlight = new Map<BatchKey, Promise<PredictionData[]>>();

const makeKey = (method: string, limit: number, horizons: number[]) =>
  `${method}|${limit}|${horizons.join(",")}`;

// （可选）持久化到 sessionStorage —— 如果不需要就删掉这三段
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

// === helpers for chart ===
function fmt(d: Date) {
  const m = d.getMonth() + 1;
  const day = d.getDate();
  return `${m.toString().padStart(2, "0")}-${day.toString().padStart(2, "0")}`;
}
function addDays(d: Date, n: number) {
  const x = new Date(d);
  x.setDate(x.getDate() + n);
  return x;
}
function sortByDateAsc<T extends { date: string | Date }>(arr: T[]) {
  return [...arr].sort((a, b) => new Date(a.date as any).getTime() - new Date(b.date as any).getTime());
}

// Best-effort builder
function buildTrendSeries(
  stock: PredictionData | null,
  extHistory7?: Array<{date: string; close: number}>
) {
  if (!stock) return [] as { label: string; price: number; isFuture: boolean }[];

  // 1) history：优先用 extHistory7；其次再看 stock.history7 / history / recentPrices
  let hist: { date: string | Date; price: number }[] = [];
  if (Array.isArray(extHistory7) && extHistory7.length) {
    const sorted = sortByDateAsc(extHistory7.map((x) => ({ date: x.date, price: x.close })));
    hist = sorted.slice(-7);
  } else {
    const anyHist = (stock as any).history7 || (stock as any).history || (stock as any).recentPrices;
    if (Array.isArray(anyHist) && anyHist.length) {
      const sorted = sortByDateAsc(
        anyHist.map((x: any) => ({ date: x.date ?? x.ts ?? x.t ?? x[0], price: x.close ?? x.price ?? x.v ?? x[1] }))
      );
      hist = sorted.slice(-7);
    } else {
      const today = new Date();
      const cur = stock.currentPrice ?? 0;
      hist = Array.from({ length: 7 }, (_, i) => ({ date: addDays(today, i - 7), price: cur }));
    }
  }

  // 2) future（保持你原逻辑）
  let fwd: { date: string | Date; price: number }[] = [];
  const anyFwd = (stock as any).forecastPath7 || (stock as any).forecast || (stock as any).future;
  if (Array.isArray(anyFwd) && anyFwd.length) {
    const sorted = sortByDateAsc(
      anyFwd.map((x: any) => ({ date: x.date ?? x.ts ?? x.t ?? x[0], price: x.price ?? x.v ?? x[1] }))
    );
    fwd = sorted.slice(0, 7);
  } else {
    const today = new Date();
    const cur = stock.currentPrice ?? 0;
    const wk = stock.predictions.find((p) => p.period === "1 Week") || stock.predictions[0];
    const target = wk ? wk.predictedPrice : cur;
    fwd = Array.from({ length: 7 }, (_, i) => {
      const t = (i + 1) / 7;
      const price = cur + (target - cur) * t;
      return { date: addDays(today, i + 1), price };
    });
  }
  if (Array.isArray(extHistory7) && extHistory7.length) {
    const sorted = sortByDateAsc(
      extHistory7.map((x: any) => ({
        date: x.date,
        price: x.close ?? x.Close ?? x.price,   // ← 兼容 Close/price
      }))
    );
    hist = sorted.slice(-7);
  }

  const merged = [...hist, { date: new Date(), price: stock.currentPrice }, ...fwd];
  const todayStr = fmt(new Date());
  return merged.map((pt) => ({
    label: fmt(new Date(pt.date)),
    price: pt.price,
    isFuture: fmt(new Date(pt.date)) > todayStr,
  }));
}

const METHOD_OPTIONS = [
  "lstm",
  "arima",
  "prophet",
  "lgbm",
  "seq2seq",
  "dilated_cnn",
  "transformer",
  "stacked",
  "ma",
  "naive-drift",
];

export const PredictTab: React.FC = () => {
  const [list, setList] = useState<PredictionData[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>(symbols[0]);
  const [selectedStock, setSelectedStock] = useState<PredictionData | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<string>("1 Day");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [method, setMethod] = useState<string>("lstm");
  const [limit, setLimit] = useState<number>(10);
  const [hist7Map, setHist7Map] = useState<Record<string, {date: string; close: number}[]>>({});

  type HistoryPoint = { date: string; close: number };
  type ChartPoint = { date: string; price?: number; pred?: number };
  const [chartErr, setChartErr] = useState<string | null>(null);
  const [history7, setHistory7] = useState<HistoryPoint[]>([]);
  const [forecastRes, setForecastRes] = useState<any | null>(null);

  // ✅ 修正：原来这一行是冗余的三元表达式，直接用 selectedSymbol 兜底即可
  const symbol = selectedSymbol || "AAPL";

  // const HORIZONS = [7, 30, 90, 180]; // ← 只要 7/30/90/180
  const HORIZONS = [1,2,3,4,5,6,7, 30]; // ← 只要 7/30/90/180
  const cacheKey = useMemo(
    () => `${method}|${limit}|${HORIZONS.join(",")}|${symbols.join(",")}`,
    [method, limit]
  );

  const [refreshNonce, setRefreshNonce] = useState(0);

  const companyName = useMemo(() => {
    const idx = symbols.indexOf(selectedSymbol);
    return companyNames[idx] ?? selectedSymbol;
  }, [selectedSymbol]);

  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true);
      setErr(null);

      // 1) 命中缓存（TTL 内）
      const cached = batchCache.get(cacheKey);
      const now = Date.now();
      if (cached && now - cached.at < CACHE_TTL_MS) {
        if (!alive) return;
        setList(cached.data);
        // 校正选中项
        const match = cached.data.find((x) => x.symbol === selectedSymbol) ?? cached.data[0] ?? null;
        setSelectedStock(match ?? null);
        if (match) {
          const periods = match.predictions.map((p) => p.period);
          if (!periods.includes(selectedPeriod)) {
            const day = match.predictions.find(p => /1\\s*day/i.test(p.period))?.period
                ?? match.predictions[0]?.period
                ?? "1 Day";
            setSelectedPeriod(day);
          }
        }
        setLoading(false);
        return;
      }

      // 2) 并发去重：复用在途请求
      let p = inFlight.get(cacheKey);
      if (!p) {
        p = (async () => {
          const lim = Number.isFinite(limit) ? Math.max(1, Math.floor(limit)) : 10;
          const raw = await fetchForecastBatch({ method, horizons: HORIZONS, limit: lim });
          const items: PredictionData[] = raw.map((r: any) => {
            const mapped = toPredictionData(r);
            return {
              ...mapped,
              method: r.method ?? method,
              lastUpdated: relativeFromNow(mapped.lastUpdated),
            } as PredictionData;
          });
          // 写缓存
          batchCache.set(cacheKey, { at: Date.now(), data: items });
          // （可选）持久化
          persistCacheToSession();
          return items;
        })();
        inFlight.set(cacheKey, p);
      }

      try {
        const items = await p;
        if (!alive) return;
        setList(items);

        const match = items.find((x) => x.symbol === selectedSymbol) ?? items[0] ?? null;
        setSelectedStock(match ?? null);

        if (match) {
          const day = match.predictions.find(p => /1\\s*day/i.test(p.period))?.period
                    ?? match.predictions[0]?.period
                    ?? "1 Day";
          setSelectedPeriod(day);
        }
      } catch (e: any) {
        if (alive) setErr(e.message || "Load failed");
      } finally {
        inFlight.delete(cacheKey);
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  // 关键依赖：method/limit 改变；手动刷新时 refreshNonce 变化绕过缓存
  }, [method, limit, refreshNonce]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setChartErr(null);
        const [h, f] = await Promise.all([fetchLast7Prices(symbol), fetchForecast7(symbol)]);
        if (!alive) return;
        setHistory7(Array.isArray(h?.prices) ? h.prices : []);
        setForecastRes(f);
      } catch (e: any) {
        if (!alive) return;
        setChartErr(e?.message ?? "Load error");
      }
    })();
    return () => {
      alive = false;
    };
  }, [symbol]);

  const currentPrediction =
    selectedStock?.predictions.find((p) => p.period === selectedPeriod) ||
    selectedStock?.predictions[0] || {
      period: "1 Day",
      predictedPrice: selectedStock?.currentPrice ?? 0,
      confidence: 0.7,
      change: 0,
      changePercent: 0,
    };

  useEffect(() => {
    let alive = true;
    (async () => {
      if (!selectedSymbol) return;
      try {
        const prices = await fetchLast7Prices(selectedSymbol);
        if (!alive) return;
        setHist7Map((m) => ({ ...m, [selectedSymbol]: prices }));
      } catch (e) {
        // 静默失败即可，图表会用 fallback
      }
    })();
    return () => { alive = false; };
  }, [selectedSymbol]);

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

  // 拼折线图数据：历史线 + 预测线
  const chartData = useMemo<ChartPoint[]>(() => {
    // 历史 7 天
    const hist: ChartPoint[] = Array.isArray(history7)
      ? history7.map((h) => ({ date: h.date, price: h.close }))
      : [];

    // “今天”的当前价（来自 selectedStock.currentPrice），只要有历史最后一天，就插入今天
    const todayISO = new Date().toISOString().slice(0, 10); // 'YYYY-MM-DD'
    const todayPoint: ChartPoint | null =
      selectedStock?.currentPrice != null && Number.isFinite(selectedStock.currentPrice)
        ? { date: todayISO, price: selectedStock.currentPrice }
        : null;

    // 预测（未来 7 天），兼容多种后端字段
    const norm = normalizeForecast(forecastRes);
    // 只取“今天之后”的 7 个点（如果后端已给了今天，则也能合并进去）
    const preds: ChartPoint[] = norm
      .map((p) => ({ date: (p.date ?? "").slice(0, 10), pred: p.value }))
      .filter((p) => p.date) // 过滤空日期
      .slice(0, 7);

    // 合并：历史7天 + 今天 + 未来7天
    const map = new Map<string, ChartPoint>();
    for (const p of hist) map.set(p.date, { ...(map.get(p.date) || {}), ...p });
    if (todayPoint) {
      const existing = map.get(todayPoint.date) || {};
      map.set(todayPoint.date, { ...existing, ...todayPoint });
    }
    for (const p of preds) {
      const existing = map.get(p.date) || {};
      map.set(p.date, { ...existing, ...p });
    }

    // 排序（按 ISO 日期）
    return Array.from(map.values()).sort((a, b) => String(a.date).localeCompare(String(b.date)));
  }, [history7, forecastRes, selectedStock?.currentPrice]);


  const trendData = useMemo(() => {
    const sym = selectedStock?.symbol || selectedSymbol;  // ← 兜底用 selectedSymbol
    const ext = sym ? hist7Map[sym] : undefined;
    return buildTrendSeries(selectedStock, ext);
  }, [selectedStock, selectedSymbol, hist7Map]);
  // 统一兼容后端各种返回结构：points / forecast / forecast_points / result / direct array
  function normalizeForecast(raw: any): Array<{ date: string; value: number }> {
    if (!raw) return [];

    // 常见结构 1：{ points: [{date, value, type}] }
    if (Array.isArray(raw.points)) {
      return raw.points
        .filter((p: any) => (p?.type ? String(p.type).toLowerCase() === "pred" : true))
        .map((p: any) => ({ date: p.date ?? p.ts ?? p.t, value: p.value ?? p.price ?? p.v }))
        .filter((p: any) => p.date && Number.isFinite(p.value));
    }

    // 常见结构 2：{ forecast: [{date, price}] }
    if (Array.isArray(raw.forecast)) {
      return raw.forecast
        .map((p: any) => ({ date: p.date ?? p.ts ?? p.t, value: p.value ?? p.price ?? p.v }))
        .filter((p: any) => p.date && Number.isFinite(p.value));
    }

    // 常见结构 3：{ forecast_points: [...] } / { result: [...] }
    if (Array.isArray(raw.forecast_points)) {
      return raw.forecast_points
        .map((p: any) => ({ date: p.date ?? p.ts ?? p.t, value: p.value ?? p.price ?? p.v }))
        .filter((p: any) => p.date && Number.isFinite(p.value));
    }
    if (Array.isArray(raw.result)) {
      return raw.result
        .map((p: any) => ({ date: p.date ?? p.ts ?? p.t, value: p.value ?? p.price ?? p.v }))
        .filter((p: any) => p.date && Number.isFinite(p.value));
    }

    // 常见结构 4：直接数组
    if (Array.isArray(raw)) {
      return raw
        .map((p: any) => ({ date: p.date ?? p.ts ?? p.t, value: p.value ?? p.price ?? p.v }))
        .filter((p: any) => p.date && Number.isFinite(p.value));
    }

    return [];
  }


  // ✅ 新增：X 轴与 Tooltip 的日期格式化（MM-DD）
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
          <h2 className="text-2xl font-bold text-gray-900">Stock Predictions</h2>
          <p className="text-gray-600">AI-powered price forecasting with confidence intervals</p>
        </div>

        {/* Controls: symbol, method, limit */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Symbol */}
          <select
            value={selectedSymbol}
              onChange={(e) => {
                const sym = e.target.value;
                setSelectedSymbol(sym);
                const match = list.find((x) => x.symbol === sym);
                if (match) setSelectedStock(match);
              }}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {(list.length ? list.map((x) => x.symbol) : symbols).map((sym) => (
              <option key={sym} value={sym}>
                {sym}
              </option>
            ))}
          </select>

          {/* Method */}
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            title="Forecast method"
          >
            {METHOD_OPTIONS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>

          {/* Limit */}
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">Limit</label>
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
              className="w-24 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          {/* ✅ 新增：刷新按钮 */}
        <button
          onClick={() => {
            batchCache.delete(cacheKey);
            setRefreshNonce((n) => n + 1);
          }}
          disabled={loading}
          className={`
            inline-flex items-center gap-2 px-3 py-2 rounded-2xl text-sm font-medium
            border border-gray-200 bg-white shadow hover:shadow-md transition
            hover:bg-gray-50 active:scale-[0.99]
            ${loading ? "opacity-60 cursor-not-allowed" : ""}
          `}
          title="Refresh cached results"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Refreshing...
            </>
          ) : (
            <>
              <RefreshCcw className="w-4 h-4" />
              Refresh
            </>
          )}
        </button>
        </div>
      </div>

      {/* 错误与加载 */}
      {err && <div className="p-3 rounded bg-red-50 border border-red-200 text-red-700">{err}</div>}
      {loading && <div className="text-sm text-gray-500">Loading...</div>}

      {/* 没数据时不渲染后续块 */}
      {!selectedStock ? null : (
        <>
          {/* Model Performance */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gradient-to-r from-blue-50 to-blue-100 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <Target className="h-5 w-5 text-blue-600" />
                <span className="text-sm font-medium text-blue-700">Model Accuracy</span>
              </div>
              <p className="text-2xl font-bold text-blue-800">{Math.round(selectedStock.accuracy * 100)}%</p>
              <p className="text-xs text-blue-600">Last 90 days</p>
            </div>

            <div className={`border rounded-lg p-4 ${getRiskColor(selectedStock.risk)}`}>
              <div className="flex items-center space-x-2 mb-2">
                <AlertCircle className="h-5 w-5" />
                <span className="text-sm font-medium">Risk Level</span>
              </div>
              <p className="text-2xl font-bold">{selectedStock.risk}</p>
              <p className="text-xs">Volatility based</p>
            </div>

            <div className="bg-gradient-to-r from-green-50 to-green-100 border border-green-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <Calendar className="h-5 w-5 text-green-600" />
                <span className="text-sm font-medium text-green-700">Best Timeframe</span>
              </div>
              <p className="text-2xl font-bold text-green-800">1-4 Weeks</p>
              <p className="text-xs text-green-600">Highest accuracy</p>
            </div>

            <div className="bg-gradient-to-r from-purple-50 to-purple-100 border border-purple-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <BarChart className="h-5 w-5 text-purple-600" />
                <span className="text-sm font-medium text-purple-700">Data Points</span>
              </div>
              <p className="text-2xl font-bold text-purple-800">2,847</p>
              <p className="text-xs text-purple-600">Updated {selectedStock.lastUpdated}</p>
            </div>
          </div>

          {/* Current Stock Info */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center">
                  <span className="text-lg font-bold text-gray-700">{selectedStock.symbol.charAt(0)}</span>
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900">{selectedStock.symbol}</h3>
                  <span className="px-2 py-0.5 text-xs rounded-full border border-blue-200 bg-blue-50 text-blue-700">
                    Method: {selectedStock.method ?? "unknown"}
                  </span>
                  <p className="text-gray-600">{selectedStock.companyName}</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-gray-900">${selectedStock.currentPrice.toFixed(2)}</p>
                <p className="text-sm text-gray-600">Current Price</p>
              </div>
            </div>

            {/* Time Period Selector */}
            <div className="flex space-x-2 mb-6">
              {selectedStock.predictions.map((pred) => (
                <button
                  key={pred.period}
                  onClick={() => setSelectedPeriod(pred.period)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedPeriod === pred.period ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  {pred.period}
                </button>
              ))}
            </div>

            {/* Prediction Details & List */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 mb-3">Price Prediction ({selectedPeriod})</h4>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-600">Predicted Price</span>
                    <span className="text-xl font-bold text-gray-900">${currentPrediction.predictedPrice.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-600">Expected Change</span>
                    <div className="flex items-center space-x-1">
                      {currentPrediction.change > 0 ? (
                        <TrendingUp className="h-4 w-4 text-green-600" />
                      ) : (
                        <TrendingDown className="h-4 w-4 text-red-600" />
                      )}
                      <span className={`font-semibold ${currentPrediction.change > 0 ? "text-green-600" : "text-red-600"}`}>
                        {currentPrediction.change > 0 ? "+" : ""}${currentPrediction.change.toFixed(2)} ({
                          currentPrediction.changePercent.toFixed(2)
                        }%)
                      </span>
                    </div>
                  </div>
                </div>

                {/* === Forecast Line Chart（历史7天 + 未来7天）=== */}
                <div className="rounded-2xl border p-4 shadow-sm">
                  <div className="flex items-center gap-2 mb-3">
                    <Calendar className="w-4 h-4" />
                    <span className="font-medium">Last 7 days & Next 7 days</span>
                  </div>
                  <div style={{ width: "100%", height: 320 }}>
                    <ResponsiveContainer>
                      <LineChart data={chartData} margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        {/* ✅ 横坐标日期改为 月-日 */}
                        <XAxis dataKey="date" tickFormatter={formatDate} tick={{ fontSize: 12 }} />
                        <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} />
                        {/* ✅ Tooltip 里也同步显示 月-日 */}
                        <Tooltip labelFormatter={(label) => `Date: ${formatDate(label as string)}`} />
                        {/* 历史线 */}
                        <Line type="monotone" dataKey="price" strokeWidth={2} dot={false} isAnimationActive={false} />
                        {/* 预测线 */}
                        <Line type="monotone" dataKey="pred" strokeWidth={2} dot isAnimationActive={false} />
                        {/* 参考线：最后一个历史点 */}
                        {history7?.length ? (
                          <ReferenceLine
                            x={history7[history7.length - 1].date}
                            strokeDasharray="3 3"
                            label={{ value: "Today", position: "top" }}
                          />
                        ) : null}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {chartErr && (
                    <div className="mt-2 text-xs text-red-600">
                      {chartErr}
                    </div>
                  )}
                </div>

              </div>

              <div className="space-y-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 mb-3">All Predictions</h4>
                  <div className="space-y-3">
                    {selectedStock.predictions.map((pred) => (
                      <div key={pred.period} className="flex items-center justify-between py-2 border-b border-gray-200 last:border-b-0">
                        <span className="text-sm text-gray-600">{pred.period}</span>
                        <div className="text-right">
                          <span className="text-sm font-semibold text-gray-900">${pred.predictedPrice.toFixed(2)}</span>
                          <div className="text-xs text-gray-500">{Math.round(pred.confidence * 100)}% confidence</div>
                        </div>
                      </div>
                    ))}
                    {!selectedStock.predictions.length && <div className="text-sm text-gray-500">No predictions yet.</div>}
                  </div>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-semibold text-blue-900 mb-2">💡 Key Insights</h4>
                  <ul className="text-sm text-blue-800 space-y-1">
                    <li>• Higher accuracy for shorter-term predictions</li>
                    <li>• Model incorporates 15+ technical indicators</li>
                    <li>• Real-time sentiment analysis included</li>
                    <li>• Risk-adjusted confidence scoring</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
