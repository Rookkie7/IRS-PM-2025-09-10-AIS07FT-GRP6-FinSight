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

async function fetchLast7Prices(symbol: string) {
  const res = await fetch(`${API}/forecast/prices7?ticker=${encodeURIComponent(symbol)}`);
  if (!res.ok) throw new Error("Failed to load last 7 prices");
  return res.json(); // { prices: Array<{ date: string; close: number }> }
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

// 未来7天预测
async function fetchForecast7(symbol: string) {
  const res = await fetch(`${API}/forecast/${encodeURIComponent(symbol)}?horizons=1,2,3,4,5,6,7`);
  if (!res.ok) throw new Error("Failed to load 7-day forecast");
  return res.json();
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
type DayBar = { date: string; close: number };
type ChartRow = { date: string; value?: number; isFuture?: boolean };

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

// ====== 组件 ======
const METHOD_OPTIONS = [
  "lstm",
  "arima",
  "prophet",
  "lgbm",
  "seq2seq",
  "dilated_cnn",
  "transformer",
  "ma",
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

  const [method, setMethod] = useState<string>("lstm");
  const [limit, setLimit] = useState<number>(10);
  const [hist7Map, setHist7Map] = useState<Record<string, HistoryPoint[]>>({});

  const [chartErr, setChartErr] = useState<string | null>(null);
  const [history7, setHistory7] = useState<HistoryPoint[]>([]);
  const [forecastRes, setForecastRes] = useState<any | null>(null);

  const symbol = selectedSymbol || "AAPL";
  const HORIZONS = [1, 2, 3, 4, 5, 6, 7, 30];
  const cacheKey = useMemo(() => `${method}|${limit}|${HORIZONS.join(",")}|${symbols.join(",")}`, [method, limit, symbols]);
  const [refreshNonce, setRefreshNonce] = useState(0);

  const companyName = useMemo(() => {
    const idx = symbols.indexOf(selectedSymbol);
    return companyNames[idx] ?? selectedSymbol;
  }, [selectedSymbol, symbols]);

  // ====== 加载推荐 + 批量预测 ======
  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true);
      setErr(null);
      try {
        // 0) 推荐 symbols（使用登录用户）
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

        // 1) /forecast/batch 只拉这些 symbols
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

        // 只保留目标 symbols（保险）
        const filtered = items.filter((x) => targetSymbols.includes(x.symbol));
        setList(filtered);

        // 默认选中
        const first = filtered[0];
        const sym0 = first?.symbol ?? targetSymbols[0] ?? "AAPL";
        setSelectedSymbol(sym0);
        setSelectedStock(first ?? null);
        if (first) {
          const day =
            first.predictions.find((p) => /1\s*day/i.test(p.period))?.period ??
            first.predictions[0]?.period ??
            "1 Day";
          setSelectedPeriod(day);
        }
      } catch (e: any) {
        if (alive) setErr(e?.message || "Load failed");
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
    // 依赖 user.id / method / limit / refreshNonce
  }, [user?.id, method, limit, refreshNonce]);

  // ====== 单只股票的历史7天与未来7天 ======
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setChartErr(null);
        const [h, f] = await Promise.all([fetchLast7Prices(symbol), fetchForecast7(symbol)]);
        if (!alive) return;
        const todayISO = new Date().toISOString().slice(0, 10);

        const hist: DayBar[] = (Array.isArray(h?.prices) ? h.prices : [])
          .map((p: any) => ({ date: String(p.date).slice(0, 10), close: Number(p.close) }))
          .filter((p: DayBar) => p.date && p.date < todayISO && Number.isFinite(p.close))
          .sort((a: DayBar, b: DayBar) => a.date.localeCompare(b.date))
          .slice(-7);

        setHistory7(hist);
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

  // 选中变化时缓存历史
  useEffect(() => {
    let alive = true;
    (async () => {
      if (!selectedSymbol) return;
      try {
        const h = await fetchLast7Prices(selectedSymbol);
        if (!alive) return;
        const todayISO = new Date().toISOString().slice(0, 10);

        const hist: DayBar[] = Array.isArray(h?.prices)
          ? (h.prices as any[])
              .map((p: any) => ({
                date: String(p.date).slice(0, 10),
                close: Number(p.close),
              }))
              .filter((p: DayBar) => p.date && p.date < todayISO && Number.isFinite(p.close))
              .sort((a: DayBar, b: DayBar) => a.date.localeCompare(b.date))
              .slice(-7)
          : [];

        setHist7Map((m) => ({ ...m, [selectedSymbol]: hist }));
      } catch {}
    })();
    return () => {
      alive = false;
    };
  }, [selectedSymbol]);

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
  const chartData = useMemo<ChartRow[]>(() => {
    // 历史（已是最近7天，不含今天），转为 {date,value}
    const histRows: ChartRow[] = Array.isArray(history7)
      ? history7.map((h) => ({ date: String(h.date).slice(0, 10), value: Number(h.close), isFuture: false }))
      : [];

    // 今天
    const todayISO = new Date().toISOString().slice(0, 10);
    const lastHistValue = histRows.length > 0 ? Number(histRows[histRows.length - 1].value) : NaN;
    const curPrice = Number.isFinite(selectedStock?.currentPrice)
      ? Number(selectedStock!.currentPrice)
      : lastHistValue;

    // 归一化未来点
    const rawPreds = normalizeForecast(forecastRes)
      .map((p) => ({ date: String(p.date).slice(0, 10), value: Number(p.value) }))
      .filter((p) => p.date && Number.isFinite(p.value));

    // 目标日期 D+1 ~ D+7
    const D = new Date(todayISO);
    const futureDates: string[] = Array.from({ length: 7 }, (_, i) => {
      const d = new Date(D);
      d.setDate(d.getDate() + (i + 1));
      return d.toISOString().slice(0, 10);
    });

    // S 曲线
    const easeCos = (t: number) => (1 - Math.cos(Math.PI * t)) / 2;

    // 生成未来逐日
    let futureDaily: Array<{ date: string; value: number }> = [];
    if (rawPreds.length >= 2 && Number.isFinite(curPrice)) {
      const anchors = [{ date: todayISO, value: curPrice as number }, ...rawPreds]
        .filter((x, i, arr) => arr.findIndex((y) => y.date === x.date) === i)
        .sort((a, b) => a.date.localeCompare(b.date));

      futureDaily = futureDates.map((dt) => {
        const rIdx = anchors.findIndex((a) => a.date >= dt);
        if (rIdx <= 0) return { date: dt, value: anchors[0].value };
        const L = anchors[rIdx - 1];
        const R = anchors[rIdx];
        if (L.date === R.date) return { date: dt, value: R.value };
        const t = (new Date(dt).getTime() - new Date(L.date).getTime()) / (new Date(R.date).getTime() - new Date(L.date).getTime());
        return { date: dt, value: Number(L.value + (R.value - L.value) * easeCos(t)) };
      });
    } else {
      const oneWeekTarget =
        selectedStock?.predictions.find((p) => /1\s*week/i.test(p.period))?.predictedPrice ??
        selectedStock?.predictions[0]?.predictedPrice ??
        curPrice;
      if (Number.isFinite(curPrice) && Number.isFinite(oneWeekTarget)) {
        futureDaily = futureDates.map((dt, i) => {
          const t = (i + 1) / 7;
          return { date: dt, value: Number(curPrice + (Number(oneWeekTarget) - Number(curPrice)) * easeCos(t)) };
        });
      }
    }

    // 合并
    const map = new Map<string, ChartRow>();
    for (const r of histRows) map.set(r.date, { ...r, isFuture: false });
    if (Number.isFinite(curPrice)) map.set(todayISO, { date: todayISO, value: Number(curPrice), isFuture: false });
    for (const p of futureDaily) map.set(p.date, { date: p.date, value: Number(p.value), isFuture: true });

    return Array.from(map.values())
      .filter((x) => Number.isFinite(x.value))
      .sort((a, b) => a.date.localeCompare(b.date));
  }, [history7, forecastRes, selectedStock?.currentPrice]);

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
          <h2 className="text-2xl font-bold text-gray-900">Stock Predictions</h2>
          <p className="text-gray-600">AI-powered price forecasting with confidence intervals</p>
        </div>

        {/* Controls */}
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

          {/* 刷新 */}
          <button
            onClick={() => {
              batchCache.delete(cacheKey);
              setRefreshNonce((n) => n + 1);
            }}
            disabled={loading}
            className={`inline-flex items-center gap-2 px-3 py-2 rounded-2xl text-sm font-medium border border-gray-200 bg-white shadow hover:shadow-md transition hover:bg-gray-50 active:scale-[0.99] ${
              loading ? "opacity-60 cursor-not-allowed" : ""
            }`}
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

      {/* 内容 */}
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

          {/* Stock Info + Chart */}
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
                <p className="text-2xl font-bold text-gray-900">
                  {Number.isFinite(selectedStock.currentPrice)
                    ? `$${Number(selectedStock.currentPrice).toFixed(2)}`
                    : "—"}
                </p>
                <p className="text-sm text-gray-600">Current Price</p>
              </div>
            </div>

            {/* Periods */}
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

            {/* Chart */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="rounded-2xl border p-4 shadow-sm">
                  <div className="flex items-center gap-2 mb-3">
                    <Calendar className="w-4 h-4" />
                    <span className="font-medium">Last 7 days & Next 7 days</span>
                  </div>

                  <div className="rounded-2xl border border-gray-100 shadow-sm p-4 bg-white">
                    <div style={{ width: "100%", height: 340 }}>
                      <ResponsiveContainer>
                        <LineChart data={chartData} margin={{ left: 12, right: 20, top: 8, bottom: 8 }}>
                          <defs>
                            <linearGradient id="priceStroke" x1="0" y1="0" x2="1" y2="0">
                              <stop offset="0%" stopColor="#3b82f6" />
                              <stop offset="100%" stopColor="#8b5cf6" />
                            </linearGradient>
                            <linearGradient id="historyFill" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="rgba(59,130,246,0.12)" />
                              <stop offset="100%" stopColor="rgba(59,130,246,0.02)" />
                            </linearGradient>
                            <linearGradient id="futureFill" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="rgba(139,92,246,0.12)" />
                              <stop offset="100%" stopColor="rgba(139,92,246,0.02)" />
                            </linearGradient>
                          </defs>

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

                          <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.4} />
                          <XAxis
                            dataKey="date"
                            tickFormatter={formatDate}
                            tick={{ fontSize: 12, fill: "#6b7280" }}
                            stroke="#e5e7eb"
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
                              Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
                            }
                            tick={{ fontSize: 12, fill: "#6b7280" }}
                            axisLine={false}
                            tickLine={false}
                            width={70}
                          />

                          <Tooltip
                            content={({ active, payload, label }) => {
                              if (!active || !payload?.length) return null;
                              const v = payload[0].payload?.value;
                              const future = payload[0].payload?.isFuture;
                              const color = future ? "#7c3aed" : "#2563eb";
                              return (
                                <div className="rounded-xl border border-gray-200 bg-white/95 backdrop-blur p-3 shadow-xl">
                                  <div className="text-xs text-gray-500 mb-1">Date: {formatDate(label as string)}</div>
                                  <div className="text-sm font-semibold" style={{ color }}>
                                    {fmtMoney.format(v ?? 0)}
                                  </div>
                                  <div className="text-[11px] text-gray-500 mt-0.5">
                                    {future ? "Forecast" : "Historical / Today"}
                                  </div>
                                </div>
                              );
                            }}
                            cursor={{ stroke: "#c7d2fe", strokeWidth: 1 }}
                          />

                          <ReferenceLine
                            x={todayISO}
                            stroke="#9ca3af"
                            strokeDasharray="3 3"
                            label={{ value: "Today", position: "top", fill: "#6b7280", fontSize: 12 }}
                          />

                          <Line type="monotone" dataKey="value" stroke="url(#priceStroke)" strokeWidth={2.5} dot={false} isAnimationActive={false} connectNulls />
                          <Line
                            type="monotone"
                            dataKey={(d: any) => (d.isFuture ? d.value : null)}
                            stroke="#7c3aed"
                            strokeWidth={2}
                            strokeDasharray="6 4"
                            dot={{ r: 3 }}
                            isAnimationActive={false}
                            connectNulls
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {chartErr && <div className="mt-2 text-xs text-red-600">{chartErr}</div>}
                </div>
              </div>

              {/* Predictions list */}
              <div className="space-y-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 mb-3">All Predictions</h4>
                  <div className="space-y-3">
                    {selectedStock.predictions.map((pred) => (
                      <div key={pred.period} className="flex items-center justify-between py-2 border-b border-gray-200 last:border-b-0">
                        <span className="text-sm text-gray-600">{pred.period}</span>
                        <div className="text-right">
                          <span className="text-sm font-semibold text-gray-900">
                            {Number.isFinite(pred.predictedPrice) ? `$${pred.predictedPrice.toFixed(2)}` : "—"}
                          </span>
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

// ====== 简易缓存（如需保留，可继续用）======
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
