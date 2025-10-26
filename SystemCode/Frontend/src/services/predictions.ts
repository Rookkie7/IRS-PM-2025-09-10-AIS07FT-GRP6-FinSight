// src/services/predictions.ts
import { fetchForecastBatchAll, fetchStockInfo, type ForecastResult } from "../api/forecast";
import { toPredictionData, type PredictionData } from "../utils/forecastMapper";

/**
 * 直接调用 /forecast/batch，随后（可选）为每个 symbol 拉公司名，并映射到 PredictionData[]
 *
 * @param symbols  指定股票（建议传入以保持顺序）；不传则由后端从库里取前 limit 个
 * @param method   预测方法（默认 'auto' 或你想要的）
 * @param horizons 多周期（默认 [7,30,90,180]）
 * @param limit    最大数量（不传则用 symbols.length 或 100）
 * @param fetchNames 是否并发请求公司名（/forecast/stock），默认 true
 */
export async function fetchBatchAsPredictionData(options: {
  symbols?: string[];
  method?: string;
  horizons?: number[];
  limit?: number;
  fetchNames?: boolean;
} = {}): Promise<PredictionData[]> {
  const {
    symbols,
    method = "auto",
    horizons = [7, 30, 90, 180],
    limit = symbols?.length ?? 100,
    fetchNames = true,
  } = options;

  // 1) 批量预测
  const batch = await fetchForecastBatchAll({
    method,
    limit,
    symbols: symbols?.join(","),
    horizons,
  });

  const results: ForecastResult[] = batch.ok.map(x => x.result);

  // 2) 并发取公司名（可选）
  let nameMap: Record<string, string> = {};
  if (fetchNames) {
    const uniq = Array.from(new Set(results.map(r => r.ticker)));
    const pairs = await Promise.allSettled(
      uniq.map(async (tk) => {
        const info = await fetchStockInfo(tk);
        const name = info?.info?.basic_info?.name || "";
        return [tk, name] as const;
      })
    );
    for (const p of pairs) {
      if (p.status === "fulfilled") {
        const [tk, name] = p.value;
        if (name) nameMap[tk] = name;
      }
    }
  }

  // 3) 映射为 PredictionData[]，尽量按 symbols 顺序输出
  const items = results.map((r) => {
    const companyName = nameMap[r.ticker] || r.ticker; // 没取到名字就用 ticker 兜底
    return toPredictionData(r, companyName);
  });

  if (symbols?.length) {
    const order = new Map(symbols.map((s, i) => [s.toUpperCase(), i]));
    items.sort((a, b) => (order.get(a.symbol.toUpperCase()) ?? 1e9) - (order.get(b.symbol.toUpperCase()) ?? 1e9));
  }

  return items;
}
