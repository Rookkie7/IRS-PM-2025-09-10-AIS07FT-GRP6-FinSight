// src/api/forecast.ts

// ---- 返回类型（与后端保持一致）----
export type ForecastPoint = {
  horizon_days: number;
  predicted: number;
  confidence?: number | null;
};

export type ForecastResult = {
  ticker: string;
  company_name?: string;    
  method: string;
  generated_at: string;
  current_price: number;
  predictions: ForecastPoint[];
};

export type BatchOkItem = {
  ticker: string;
  result: ForecastResult;     // svc.forecast 的返回
};

export type BatchFailItem = {
  ticker: string;
  error: string;
};

export type BatchResponse = {
  requested: number;
  succeeded: number;
  failed: number;
  ok: BatchOkItem[];
  fail: BatchFailItem[];
};

// ---- 基础地址（可用 .env 配置）----
const BASE = import.meta.env.VITE_BACKEND_BASE_URL ?? "http://127.0.0.1:8000";

/**
 * 直接调用批量路由 /forecast/batch
 *
 * @param options.method        预测方法，如 'naive-drift' | 'auto' | 'transformer' | 'ensemble(a,b,...)'
 * @param options.limit         最多处理多少只股票（默认 100）
 * @param options.symbols       逗号分隔的代码，如 'AAPL,TSLA'；不传则从库里读取前 limit 个
 * @param options.horizons      多周期，如 [7,30,90,180]；不传则走后端默认
 * @param options.horizon_days  单一周期；与 horizons 二选一
 * @param options.concurrency   并发度（默认 8）
 */
export async function fetchForecastBatchAll(options: {
  method?: string;
  limit?: number;
  symbols?: string;       // e.g. "AAPL,TSLA,NVDA"
  horizons?: number[];
  horizon_days?: number;
  concurrency?: number;
} = {}): Promise<BatchResponse> {
  const qs = new URLSearchParams();

  if (options.method) qs.set("method", options.method);
  if (options.limit != null) qs.set("limit", String(options.limit));
  if (options.symbols) qs.set("symbols", options.symbols);
  if (options.horizons && options.horizons.length) {
    qs.set("horizons", options.horizons.join(","));
  } else if (options.horizon_days != null) {
    qs.set("horizon_days", String(options.horizon_days));
  }
  if (options.concurrency != null) qs.set("concurrency", String(options.concurrency));

  const url = `${BASE}/forecast/batch?${qs.toString()}`;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`batch ${res.status}: ${text || res.statusText}`);
  }

  const data = (await res.json()) as BatchResponse;

  // 轻校验关键字段，便于在控制台快速发现后端字段改动
  if (!data || !Array.isArray(data.ok) || !Array.isArray(data.fail)) {
    throw new Error("Unexpected /forecast/batch response shape");
  }
  return data;
}

/**
 * 便捷方法：只取成功结果里的 ForecastResult 数组
 */
export async function fetchForecastBatchResultsOnly(options?: Parameters<typeof fetchForecastBatchAll>[0]) {
  const r = await fetchForecastBatchAll(options);
  return r.ok.map(x => x.result);
}



/** ✅ 单只预测：/forecast?ticker=...&horizon_days=...&horizons=...&method=... */
export async function fetchForecast(
  ticker: string,
  horizonsOrHorizon: number[] | number = 7,
  method:
    | "naive-drift" | "ma" | "arima" | "prophet" | "lgbm" | "lstm"
    | "seq2seq" | "dilated_cnn" | "transformer" | "stacked" | "auto"
    | `ensemble(${string})` = "naive-drift"
): Promise<ForecastResult> {
  const qs = new URLSearchParams();
  qs.set("ticker", ticker);
  if (Array.isArray(horizonsOrHorizon)) {
    const hs = horizonsOrHorizon.filter((x) => x > 0);
    if (hs.length) qs.set("horizons", hs.join(","));
  } else {
    qs.set("horizon_days", String(horizonsOrHorizon));
  }
  qs.set("method", method);
  const url = `${BASE}/forecast?${qs.toString()}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`forecast error ${res.status}: ${await res.text()}`);
  return res.json();
}

/** （可选）批量：/forecast/batch?method=...&limit=...&symbols=... */
export async function fetchForecastBatch(params: {
  method?: string;
  limit?: number;
  symbolsCsv?: string;   // "AAPL,TSLA"
  horizons?: number[];
  horizon_days?: number;
  concurrency?: number;
} = {}) {
  const qs = new URLSearchParams();
  if (params.method) qs.set("method", params.method);
  if (params.limit != null) qs.set("limit", String(params.limit));
  if (params.symbolsCsv) qs.set("symbols", params.symbolsCsv);
  if (params.horizons?.length) qs.set("horizons", params.horizons.join(","));
  else if (params.horizon_days != null) qs.set("horizon_days", String(params.horizon_days));
  if (params.concurrency != null) qs.set("concurrency", String(params.concurrency));

  const res = await fetch(`${BASE}/forecast/batch?${qs.toString()}`);
  if (!res.ok) throw new Error(`batch error ${res.status}: ${await res.text()}`);

  const data = await res.json(); // { ok:[{ticker,result}], fail:[...] }
  if (Array.isArray(data?.ok)) {
    return data.ok.map((e: any) => e.result as ForecastResult);
  }
  throw new Error("Unexpected /forecast/batch response");
}

/** 单只股票基础信息（拿公司名等）：/forecast/stock?ticker= */
export type StockInfoResponse = {
  symbol: string;
  info: {
    basic_info?: { name?: string; [k: string]: unknown };
    [k: string]: unknown;
  };
};

export async function fetchStockInfo(ticker: string): Promise<StockInfoResponse> {
  const url = `${BASE}/forecast/stock?ticker=${encodeURIComponent(ticker)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`stock ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function fetchLast7Prices(ticker: string): Promise<Array<{date: string; close: number}>> {
  // 若你的后端有全局前缀，请改成 /api/v1/forecast/prices7
  const res = await fetch(`/forecast/prices7?ticker=${encodeURIComponent(ticker)}`);
  if (!res.ok) throw new Error(`prices7 http ${res.status}`);
  const json = await res.json();
  return Array.isArray(json?.prices) ? json.prices : [];
}